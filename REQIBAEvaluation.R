##Have the UAV pick which candidate BS to connect to using REQIBA as it is travelling through an urban environment
#Online training of the DQN block
rm(list=ls())
library(spatstat)
library(VGAM)
library(hypergeo)
library(keras)
library(tensorflow)
library(parallel)
library(matlab)
source("MiscFunctions.R")



#number of MC trials, episodes and steps
MCtrials = 1000
episodes = 1
steps = 100


DISCOUNT = 0.1
REPLAY_MEMORY_SIZE = 10000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 10000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 4096/2  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

#how many candidate BSs to consider, and how many interfering BSs to consider per candidate
BScand = 10
BScandi = 125

handoverpenalty = 0.5

#which BS to be connected to
whichBS = 1


alpha = 0.5
beta = 300
gamma = 20


#building parameters
buildDens = 300/(1000^2)
buildWidth = 40
buildR = buildWidth/(2*sin(pi/4))
heightParam = 20

#number of antenna elements
Nt= 8

#max and min UAV heights
minH = 0
maxH = 300

velocity = 10

#Base station height
BSh = 30
windowWidth = 5000

#Base station density
BHdensity = 5/(1000^2)

#UAV antenna beamwidth in radians
UAVBHBW = pi*1/4

al = 2.1 #pathloss exponent under LoS
an = 4 #pathloss exponent under NLOS
mal = 10 #Nakagami-m multipath fading under LOS (unused)
man = 1 #Nakagami-m multipath fading under NLOS (unused)
BStx = 40 #BS transmit power in Watt

#transmit frequency
Freq = 2*10^9

#resource block beamwidth (unused)
RBbeamwidth = 180000

#noise power
N = -174+10*log10(20*10^6)+10
N = 10^(N/10)/1000

#BS antenna tilt
BStilt=-10
BHtilt=-10

#allow multicore support
cores=12L

k_clear_session()
threads <- cores
config <- tf$compat$v1$ConfigProto(intra_op_parallelism_threads = threads, 
                                   inter_op_parallelism_threads = threads)
session = tf$compat$v1$Session(config=config)
tf$compat$v1$keras$backend$set_session(session)


#Raytracing function for checking LOS
isLOS = function(buildings,buildR,buildH,x0,y0,x,y,h,BSh){
  angle = atan2((y-y0),(x-x0))
  dist = sqrt((x-x0)^2+(y-y0)^2)
  
  build = buildings
  build = shift(build,c(-x0,-y0))
  build = rotate(build,angle=-angle)
  
  buildX = build$x
  buildY = build$y
  foo = which(buildX<dist)
  buildX = buildX[foo]
  buildY = buildY[foo]
  buildH = buildH[foo]
  
  foo = which(buildX>0)
  buildX = buildX[foo]
  buildY = buildY[foo]
  buildH = buildH[foo]
  
  foo = which(abs(buildY)<=buildR)
  buildX = buildX[foo]
  buildY = buildY[foo]
  buildH = buildH[foo]
  
  foo = buildH>((abs(h-BSh)*(buildX/dist)+min(BSh,h)))
  if(length(which(foo==TRUE))>0){
    return(FALSE)
  }
  else{
    return(TRUE)
  }
}


antennaGain = function(r,BSh,h,Nt){
  angle = atan2(BSh-h,r)
  g=(1/Nt)*((sin(Nt*pi*(sin(angle))/2)^2)/(sin(pi*(sin(angle))/2)^2))
  g=g*BStx*K
  return(g)
}


maxSINR = 10.45007
#normalise the neural network input data
normaliseData = function(P,Int,Dist,angle,h,which,BSIDs,load,meanDist,step){
  P= log10(P)/10  
#  }
  
  Dist = Dist/10
  meanDist = meanDist/10
  Int = log10(Int)/10
  Int[is.infinite(Int)]=-20
  
  angle=angle/(pi)
  hotcode = vector(length=BScand)
  hotcode[]=0
  hotcode[which(which==BSIDs)]=1
  
  h = h/300
  
  load = load/100
  
  if(step==(steps-1)){
    terminal=1
  }else{terminal=0}
  
  return(c(P,Int,h,hotcode,terminal))
}

inputNum = 3*BScand+2

#DDQN
try(k_constant(1), silent=TRUE)
try(k_constant(1), silent=TRUE)
options(bitmapType='cairo')

#two initial layers
main_input <- layer_input(shape = c(inputNum),name='input_layer')

dense <- main_input %>% 
  layer_dense(units = inputNum, activation = 'linear',name='dense1')

#value function stream
value_function <- dense %>% 
  layer_dense(units = inputNum, activation = 'linear',name="value_dense")%>% 
  layer_dense(units=1, activation = 'relu',name="value_output")

#advantage function stream
advantage_function <- dense %>% 
  layer_dense(units = inputNum, activation = 'linear',name="adv_dense")%>% 
  layer_dense(units=BScand, activation = 'relu',name="adv_output")

#merging the two streams
combine <- layer_lambda(name="Dueling_DQN",f=function(x){
  return(x[[1]]-k_mean(x[[1]],keepdims=TRUE)+x[[2]])
})(c(advantage_function,value_function))

model <- keras_model(
  inputs = c(main_input), 
  outputs = c(combine)
)

model %>% compile(
  optimizer = 'adam', 
  loss = 'mse',
  metrics = c('accuracy')
)


#IPNN
regressionModel <- keras_model_sequential()
regressionModel %>%
  layer_dense(units=10,input_shape = 3) %>%
  layer_dense(units = 20, activation = 'linear') %>%
  layer_dense(units = 20, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'linear')



regressionModel %>% compile(
  optimizer = 'adamax', 
  loss = 'mse',
  metrics = c('mae')
)

regressionModel %>% load_model_weights_hdf5("IPNNweights.h5")


#create target model
targetModel = model
targetModel  %>% set_weights(model %>% get_weights())


#do online training/evaluation over a range of UAV heights
h = seq(from=20,to=200,by=180/10)

#our performance metric. We measure the the total data throughput (per unit bandwidth) normalised over the episode throughput achieved by the closest-BS association
#we also store the handover rate per min
achieveableRate = zeros(nrow=MCtrials*episodes,ncol=length(h))
oSINRAchieveableRate = zeros(nrow=MCtrials*episodes,ncol=length(h)) #achieveable rate when associated based on omnidirectional SINR
closestAchieveableRate = zeros(nrow=MCtrials*episodes,ncol=length(h)) #achieveable rate when associated based on closest BS

#the handover rates
handovers = zeros(nrow=MCtrials*episodes,ncol=length(h))
oSINRHandovers = zeros(nrow=MCtrials*episodes,ncol=length(h))
closestHandovers = zeros(nrow=MCtrials*episodes,ncol=length(h))


##UAV antenna gain
UAVBHgain = 4*pi/((UAVBHBW/2)^2)

#near-field pathloss
K = (((3*10^8)/Freq)/(4*pi))^2



#multiple MC trials. In each trial we generate the environment and then run the algorithm over multiple episodes
for(l in 5:length(h)){
epsilon = 1  

#the replay memory for the training
current_state_replay = zeros(nrow=REPLAY_MEMORY_SIZE,ncol=inputNum)
action_replay = vector(length=REPLAY_MEMORY_SIZE)
reward_replay = vector(length=REPLAY_MEMORY_SIZE)
next_state_replay = zeros(nrow=REPLAY_MEMORY_SIZE,ncol=inputNum)
replay_index = 1

  
  
for(m in 1:MCtrials){
    print(m)

    cPm= vector(length=MCtrials)
    cPmn = vector(length=MCtrials)
    cPmML= vector(length=MCtrials)
    
  #  whichNthBS = vector(length=MCtrials)
    
    #uav travel vector
    uavx= seq(from=-(velocity*steps/2),to=(velocity*steps/2),by=velocity)
    uavy=0
    
    num = vector(length=MCtrials)

     UAVBHgain = 4*pi/((UAVBHBW/2)^2)

      ##window in which we simulate BS distribution
      sWindow = owin(xrange=c(-windowWidth/2,windowWidth/2),yrange=c(-windowWidth/2,windowWidth/2))
      gen=FALSE
      while(gen==FALSE){
        BHppp = rpoispp(lambda=BHdensity,win=sWindow)
        if(BHppp$n>BScand){
          gen=TRUE
        }
      }
      
      buildings = gridcenters(window=sWindow,nx=floor(sqrt(buildDens*(5000^2))),ny=ceil(sqrt(buildDens*(5000^2))))#rpoispp(lambda=buildDens,win=windowBS)
      build = rpoispp(lambda=buildDens,win=sWindow)
      build$x = buildings$x
      build$y = buildings$y
      build$n = length(buildings$x)
      buildings = build
      buildH = rrayleigh(n=buildings$n,scale=heightParam)
      
      BSload = rexp(n=BHppp$n,rate=0.05)#rnorm(n=BHppp$n,mean=50,sd=15)
      BSload[]=1
      
      ##which network a given BS belongs to
      ##Note, I wrote this code for multi-network scenarios where the UAV can choose from several different operator networks
      ##We decided against looking at multi-network scenarios for the TVT paper
      ##Nonetheless the dataset and the resulting neural network is capable of distinguishing between different networks, we simply don't use that function in the simulations
      whichNetwork=ones(nrow=BHppp$n,ncol=1)#floor(runif(n=BHppp$n,min=1,max=4))
  
     #generate the state spaces for all of the steps in the MC trial
     #the iteration function generates all of the state observations a priori, as well as the rewards for each of the possible actions
     #then we have the main loop below which goes through the states, takes actions, and gets rewards
     #we separate out the state generation and the decision loop because it seems to be more computationally efficient to call this loop in one go and generate the entire episode worth of data, using parallel threads
      iteration = function(q){
        j = floor(q/steps)+1
        w = q%%steps+1
        LOS = vector(length=BHppp$n)
        
        measuredpower = vector(length=BHppp$n)
      #  connectedto=floor(runif(n=1,min=1,max=(BScand+1))) #the UAV is connected to one of the BScand closest BSs (so it then decides whether to stay connected or change)
        numInterferers = zeros(nrow=BScand,ncol=BScandi)
        intLOS = zeros(nrow=BScand,ncol=BScandi)
        intP = zeros(nrow=BScand,ncol=BScandi)
        distances = vector(length=BScand)
        meanDistances = vector(length=BScand)
        cov = vector(length=BScand)
        
        
        #for each BS get the distance, channel type (LOS/NLOS) and the signal power received by the omnidirectional antenna
        rdist = vector(length=BHppp$n)
        mdist = vector(length=BHppp$n)
        load = vector(length=BHppp$n)
        for(i in 1:BHppp$n){
          load[i] = ceil(min(100,max(0,BSload[i])))
          if(isLOS(buildings=buildings,buildR=buildR,buildH=buildH,x0=uavx[w],y0=0,x=BHppp$x[i],y=BHppp$y[i],h=h[l],BSh=BSh)){
            LOS[i]=TRUE
          }
          else{LOS[i]=FALSE
          }
          
          angle = (atan2(BSh-h[l],sqrt((BHppp$x[i]-uavx[w])^2+(BHppp$y[i])^2)))
          rdist[i] = (sqrt((BHppp$x[i]-uavx[w])^2+(BHppp$y[i])^2))
          z = w:steps
          mdist[i] = mean(sqrt((BHppp$x[i]-uavx[z])^2+(BHppp$y[i])^2))/1000
          g=(1/Nt)*((sin(Nt*pi*(sin(angle))/2)^2)/(sin(pi*(sin(angle))/2)^2))
          g = g*BStx*K
          
          if(LOS[i]==TRUE){
            measuredpower[i]=g*(sqrt((BHppp$x[i]-uavx[w])^2+(BHppp$y[i])^2+(BSh-h[l])^2))^(-al)  
          }else{measuredpower[i]=g*(sqrt((BHppp$x[i]-uavx[w])^2+(BHppp$y[i])^2+(BSh-h[l])^2))^(-an)}
        }
        
        #now get the omnidirectional SINR for each BS
        oSINR = vector(length=BHppp$n)
        for(i in 1:BHppp$n){
          foo = 1:BHppp$n
          foo = foo[foo!=i]
          oSINR[i] = measuredpower[i]/(sum(measuredpower[foo])+N)
        }
        
        
        iBSBHHeight = vector(length=BHppp$n)
        iBSBHHeight[LOS==TRUE]=0
        iBSBHHeight[LOS==FALSE]=Inf
        order1 =order(rdist,decreasing=FALSE) #get the order of BSs according to distance to UAV
        order2 =order(measuredpower,decreasing=TRUE) #get the order of BSs according to omni SINR
        order = c(order1[1:(BScand/2)],order2[1:(BScand/2)]) #get the list of candidate BSs to choose from
        
        
        achieveableRate = vector(length=BScand)
        SINR = vector(length=BScand)
       # RBsneeded = vector(length=BScand)
        angleVector = vector(length=BScand)
        
        
        ##get the rate that would be achieved from each candidate BS through the directional antenna
        for(i in 1:BScand){
          BHdist = sqrt((BHppp$x-uavx[w])^2+(BHppp$y)^2)
          BHBS = c(BHppp$x[order[i]],BHppp$y[order[i]])
          
          angleVector[i] = acos((1*BHBS[1]+0*BHBS[2])/(sqrt(1^2+0^2)*sqrt(BHBS[1]^2+BHBS[2]^2)))
    
          ind = order[i]
          BHint = 1:BHppp$n
          
          
          BHint = BHint[BHint!=ind]
          BSdist = BHdist[ind]
          BHdist = BHdist[BHint]
          
          distances[i]=BSdist/1000
          
          hopt = h[l]
          angle = atan2(hopt-BSh,BSdist)
          
          #exclude the BSs that are outside the antenna radiation lobe
          if(UAVBHBW<pi/2){
            if((angle < (pi/2-(UAVBHBW/2))) && (angle > (UAVBHBW/2))){
              uthreshold = (hopt-BSh)/tan(angle-(UAVBHBW/2))  
            }
            else if(angle>(pi/2-(UAVBHBW/2))){
              uthreshold = (hopt-BSh)/tan(pi/2 - UAVBHBW)  
            }
            else{
              uthreshold = windowWidth
            }
          }
          else{uthreshold = windowWidth}
          
          if(angle<(pi/2-UAVBHBW/2)){
            lthreshold = (hopt-BSh)/tan(angle+(UAVBHBW/2))  
          }else{lthreshold=0}
          
          BHint = BHint[find(BHdist<=uthreshold)]
          BHdist = BHdist[BHdist<=uthreshold]
          BHint = BHint[find(BHdist>=lthreshold)]
          BHdist = BHdist[BHdist>=lthreshold]
          
          BHint = getInt2(x=c(uavx[w],0),int=BHint,BHBS=BHBS,grid=BHppp,UAVBHBW=UAVBHBW)
          BHH=iBSBHHeight[BHint]
          BHLOS = LOS[BHint]
          estpower = measuredpower[BHint]
          BHint = cbind(BHppp$x[BHint],BHppp$y[BHint])
          
          #for the BHint closest interfering BSs, store the BS powers received by the omnidirectional antenna, the distance and the channel type
          if(length(BHint)>0){
            iD = sqrt((BHint[,1]-uavx[w])^2+(BHint[,2])^2)
            f = order(iD,decreasing=FALSE)
            for(k in 1:min(BScandi,length(f))){
              numInterferers[i,k] = iD[f[k]]
              intP[i,k] = estpower[f[k]]
              if(BHLOS[f[k]]==TRUE){intLOS[i,k]=1}
              else{intLOS[i,k]=0}
            }
          }
          

          
          ##get spectral efficiency
          specEff = getDualRateRamyAntenna(x=c(uavx[w],0,hopt),BHBS=BHBS,BSh=BSh,withFading=FALSE,LOS=LOS[ind],iBSBH=cbind(BHint[,1],BHint[,2]),BHtilt=BHtilt,iBSBHHeight=BHH,Nt=Nt,al=al,an=an,mal=mal,man=man,PWRgain=BStx*UAVBHgain*K,N=N,alpha=alpha,beta=beta,gamma=gamma)
      #    RBsneeded[i] = 0 #not using this
          SINR[i]=2^(specEff)-1 #SINR during the timestep, with the multipath fading effects ignored due to the timestep duration
          specEff = getDualRateRamyAntenna(x=c(uavx[w],0,hopt),BHBS=BHBS,BSh=BSh,withFading=TRUE,LOS=LOS[ind],iBSBH=cbind(BHint[,1],BHint[,2]),BHtilt=BHtilt,iBSBHHeight=BHH,Nt=Nt,al=al,an=an,mal=mal,man=man,PWRgain=BStx*UAVBHgain*K,N=N,alpha=alpha,beta=beta,gamma=gamma)
        #  RBsneeded[i] = 0#ceil(minCCdatarate/(200000*specEff)) #not using this
          cov[i]=2^(specEff)-1 #instantaneous SINR value, including the multipath fading effects (We would use this if we were interested in the coverage probability)
        }
        
        
        order1 =order(rdist,decreasing=FALSE)
        order2 =order(measuredpower,decreasing=TRUE)
        order = c(order1[1:(BScand/2)],order2[1:(BScand/2)])
        
        foo = 1:BScand
        
        return(c(measuredpower[order[foo]],as.vector(t(numInterferers)),distances[foo],angleVector[foo],cov[foo],SINR[foo],order[foo],load[order[foo]],mdist[order[foo]],as.vector(t(intLOS)),as.vector(t(intP))))
      }
      
      #generate the environmental observations across all of the steps and for each possible UAV-BS association, using multi-threaded process
      X=0:(steps-1)
      opt = mclapply(X=X,FUN=iteration,mc.cores=cores)
      
      mP = 1:(BScand)
      Int = (mP[BScand]+1):(mP[BScand]+BScand*BScandi)
      Distances = (Int[BScand*BScandi]+1):(Int[BScand*BScandi]+BScand)
      Angles = (Distances[BScand]+1):(Distances[BScand]+BScand)
      C = (Angles[BScand]+1):(Angles[BScand]+BScand)
      sSINR = (C[BScand]+1):(C[BScand]+BScand)
      BSID = (sSINR[BScand]+1):(sSINR[BScand]+BScand)
      load = (BSID[BScand]+1):(BSID[BScand]+BScand)
      meanDistances = (load[BScand]+1):(load[BScand]+BScand)
      IntLOS = (meanDistances[BScand]+1):(meanDistances[BScand]+BScand*BScandi)
      IntP = (IntLOS[BScand*BScandi]+1):(IntLOS[BScand*BScandi]+BScand*BScandi)
    
      
      mcS =array(dim=c(1,steps,BScand))
      mcInt =array(dim=c(1,steps,BScand*BScandi))
      mcDist =array(dim=c(1,steps,BScand))
      mcmeanDist =array(dim=c(1,steps,BScand))
      mcAngles =array(dim=c(1,steps,BScand))
      mcCov =array(dim=c(1,steps,BScand))
      mcSINR =array(dim=c(1,steps,BScand))
      mcBSID =array(dim=c(1,steps,BScand))
      mcLoad =array(dim=c(1,steps,BScand))
      mcIntLOS =array(dim=c(1,steps,BScand*BScandi))
      mcIntP =array(dim=c(1,steps,BScand*BScandi))
      
      #extract and store the data we generated in the iteration loop
      for(k in 0:(steps-1)){
        j = floor(k/steps)+1
        w = k%%steps+1
        mcS[j,w,]=opt[[k+1]][mP]
        mcInt[j,w,]=opt[[k+1]][Int]
        mcDist[j,w,]=opt[[k+1]][Distances]
        mcAngles[j,w,]=opt[[k+1]][Angles]
        mcCov[j,w,] =opt[[k+1]][C]
        mcSINR[j,w,] =opt[[k+1]][sSINR]
        mcBSID[j,w,]=opt[[k+1]][BSID]
        mcLoad[j,w,]=opt[[k+1]][load]
        mcmeanDist[j,w,]=opt[[k+1]][meanDistances]
        mcIntLOS[j,w,]=opt[[k+1]][IntLOS]
        mcIntP[j,w,]=opt[[k+1]][IntP]
      }
      
      #apply the IPNN model to predict interference powers from the interference distances, channel types and UAV heights for all steps in the episode
      fooP = as.vector(t(mcIntP[1,,]))
      foo = as.vector(t(mcInt[1,,]))/1000
      fooL = as.vector(t(mcIntLOS[1,,]))
      fooH = vector(length=length(fooL))
      fooH[] = h[l]/300
      check=which(foo==0)
      reg = regressionModel %>% predict(cbind(foo,fooL,fooH))
      g=10^(reg)
      g[check]=0
      rInt = matrix(g,nrow=steps,ncol=BScand*BScandi,byrow=TRUE)
    

      
      
      #which BS is has strongest omni-directional signal
      optAss=vector(length=steps)
      for(i in 1:steps){
        optAss[i] = which.max(mcS[1,i,])  
      } 
      
      #for each episode in a MC trial (by default, there is only 1 episode per MC trial, we can do several episodes per MC trial if we want to have the UAV travel the same path through the same environment multiple times)
      for(k in ((m-1)*episodes+1):(m*episodes)){
      #starting height
      UAVh = h[l]
      whichBS = floor(runif(n=1,min=1,max=(BScand+1))) #for the first step in the episode, assume we are initially associated with one of the candidate BSs at random
      UAVass = mcBSID[1,1,whichBS]
      isHandover = FALSE
      SINRisHandover=FALSE
      closestisHandover=FALSE
      closestUAVh = h[l]
      closestUAVass = mcBSID[1,1,whichBS]
      SINRUAVh = h[l]
      SINRUAVass = mcBSID[1,1,whichBS]
      cat("Height: ",l,"MC Trial: ",m," Episode: ",k,"\n")
      episode_reward = 0
      episode_handovers = 0
      closest_episode_reward=0
      closest_episode_handovers=0
   #   SINRH_episode_reward=0
      SINR_episode_reward=0
      SINR_episode_handovers = 0
      lastaction = whichBS #what was the action taken in the previous step? (random, for the first step in the episode)
      
      #for each step
      for(j in 1:(steps-1)){
     #   print(j)
        foo = 1
        
        #get the output of the IPNN for the current step
        rrInt = vector(length=BScand)
        for(i in 1:BScand){
        rrInt[i] = sum(rInt[j,((i-1)*BScandi+1):(i*BScandi)])+N
        }
      
        #normalise the current state
        current_state = normaliseData(P=mcS[foo,j,],Int=rrInt,Dist=mcDist[foo,j,],angle = mcAngles[foo,j,],h=h[l],which=mcBSID[foo,j,lastaction],BSIDs=mcBSID[foo,j,],load=mcLoad[foo,j,],meanDist=mcmeanDist[foo,j,],step=j)
        foo = rbind(current_state[1:inputNum],zeros(nrow=(MINIBATCH_SIZE-1),ncol=inputNum))
        
        #take an action
        #action is the index of the BS to associate with for the timestep
        
        #if action not taken at random, choose best action based on DDQN output
        if(runif(n=1,min=0,max=1)>epsilon){
        action = model %>% predict(foo)
        action = which.max(action[1,])
        }else{#otherwise, pick action at random
        action = floor(runif(n=1,min=1,max=(BScand+1)))
        }

        #check if a handover occurs
        if(UAVass!=mcBSID[1,j,action]){
        isHandover=TRUE
        episode_handovers=episode_handovers+1
        }
        
        UAVass = mcBSID[1,j,action]
  
        #closest association    
        closestaction= which.min(mcDist[1,j,])
        if(mcBSID[1,j,closestaction]!=closestUAVass){
        closestisHandover=TRUE
        closest_episode_handovers=closest_episode_handovers+1
        }
        closestUAVass = mcBSID[1,j,1]
   
        #strongest omni-directional SINR association
        if(SINRUAVass!=mcBSID[1,j,optAss[j+1]]){
        SINRisHandover=TRUE
        SINR_episode_handovers=SINR_episode_handovers+1
        }
        SINRUAVass = mcBSID[1,j,optAss[j+1]]
        
        #get the new state and reward

        rrInt = vector(length=BScand)
        for(i in 1:BScand){
          rrInt[i] = sum(rInt[j+1,((i-1)*BScandi+1):(i*BScandi)])+N
        }
        
        foo = 1
        new_state = normaliseData(P=mcS[foo,j+1,],Int=rrInt,Dist=mcDist[foo,j+1,],angle = mcAngles[foo,j+1,],h=h[l],which=mcBSID[foo,j+1,action],BSIDs=mcBSID[foo,j+1,],load=mcLoad[foo,j+1,],meanDist=mcmeanDist[foo,j+1,],step=(j+1))
        
        #get the reward values (the throughput per unit bandwidth)
        if(isHandover==FALSE){
        reward = RBbeamwidth*mcLoad[foo,j,action]*log2(1+mcSINR[foo,j,action])/(10^6)
        }
        else{reward = handoverpenalty*RBbeamwidth*mcLoad[foo,j,action]*log2(1+mcSINR[foo,j,action])/(10^6)}
        
        foo = 1
          if(closestisHandover==FALSE){
          closestReward = RBbeamwidth*mcLoad[foo,j,closestaction]*log2(1+mcSINR[foo,j,closestaction])/(10^6)
          }
          else{ closestReward =  handoverpenalty*RBbeamwidth*mcLoad[foo,j,closestaction]*log2(1+mcSINR[foo,j,closestaction])/(10^6)}
        
        foo = 1

        if(SINRisHandover==FALSE){
        SINRReward= RBbeamwidth*mcLoad[foo,j,optAss[j]]*log2(1+mcSINR[1,j,optAss[j]])/(10^6)
        }else{SINRReward= handoverpenalty*RBbeamwidth*mcLoad[foo,j,optAss[j]]*log2(1+mcSINR[1,j,optAss[j]])/(10^6)}
        
        #reset the handover flags and update the action
        isHandover=FALSE
        closestisHandover=FALSE
        SINRisHandover=FALSE
        
        lastaction=action
        
        ##add to the replay memory
        if(replay_index<REPLAY_MEMORY_SIZE){
        current_state_replay[replay_index,]=current_state
        action_replay[replay_index]=action
        reward_replay[replay_index]=reward
        next_state_replay[replay_index,]=new_state
        replay_index = replay_index+1
        }else{
          current_state_replay=rbind(current_state_replay[2:REPLAY_MEMORY_SIZE,],current_state)
          action_replay = c(action_replay[2:REPLAY_MEMORY_SIZE],action)
          reward_replay = c(reward_replay[2:REPLAY_MEMORY_SIZE],reward)
          next_state_replay=rbind(next_state_replay[2:REPLAY_MEMORY_SIZE,],new_state)
        }
        
        ##training
        if(replay_index>=MIN_REPLAY_MEMORY_SIZE){
          
        #randomly sample from the replay memory for a minibatch of data  
        minibatch = sample(1:replay_index,size=MINIBATCH_SIZE,replace=FALSE)
        curr = current_state_replay[minibatch,]
        act = action_replay[minibatch]
        rev = reward_replay[minibatch]
        future = next_state_replay[minibatch,]
        
        #get the Q values for the current states from the neural network
        curr_q_list =  model %>% predict(curr)
        
        #get the best action for each future state using the neural network
        future_q_select = model  %>% predict(future)
        future_q_select = apply(future_q_select,1,which.max)
  
        #get the Q values for the best action (as predicted above) for each of the future states using the target DQN
        future_q_list = targetModel  %>% predict(future)
        
        X = zeros(nrow=MINIBATCH_SIZE,ncol=inputNum)
        Y = zeros(nrow=MINIBATCH_SIZE,ncol=BScand)
        
        #for each entry in the minibatch do training
        for(i in 1:MINIBATCH_SIZE){
          
        #max Q value from the future states  
        max_future_Q = future_q_list[i,future_q_select[i]]
        
        #if not terminal state
        if(curr[i,inputNum]==0){
        new_Q = rev[i]+DISCOUNT*max_future_Q
        }else{new_Q = rev[i]}
        #Q values for the current state
        Qs = curr_q_list[i,]
        #update the Q value for the action taken
        Qs[act[i]]=new_Q
          X[i,]=curr[i,]
        Y[i,]=Qs
        }
        #fit the 
        model  %>% fit(X, Y,batch_size=MINIBATCH_SIZE,verbose=0,epochs=20)
        }
        
        #update cumulative rewards
        episode_reward=reward+episode_reward
        closest_episode_reward=closestReward+closest_episode_reward
        SINR_episode_reward = SINRReward+ SINR_episode_reward 
        
      }
      
  
      achieveableRate[k,l] = episode_reward
      closestAchieveableRate[k,l] = closest_episode_reward
      oSINRAchieveableRate[k,l] = SINR_episode_reward
      handovers[k,l] =episode_handovers
      closestHandovers[k,l] = closest_episode_handovers
      oSINRHandovers[k,l] = SINR_episode_handovers
    
      
      if(k%%5==0){
      #once every 5 episodes update the targetModel to have the same weights as the model
      targetModel  %>% set_weights(model %>% get_weights())
      }
      
      #decay the exploration factor
      if(epsilon > MIN_EPSILON){
        epsilon = epsilon*EPSILON_DECAY
      epsilon = max(MIN_EPSILON, epsilon)
      }
      
      #plot the training process
      plot(x=1:(k),y=oSINRAchieveableRate[1:k,l]/closestAchieveableRate[1:k,l],col='red',type='l',xlab="steps",ylab="Achieveable Rate (bps/hz)",ylim=c(0,3))
      lines(x=1:(k),y=closestAchieveableRate[1:k,l]/closestAchieveableRate[1:k,l],col='blue')
      lines(x=1:(k),y=achieveableRate[1:k,l]/closestAchieveableRate[1:k,l])
      
      }
      

      #plot the main figures
      #  achm = mean(achieveableRate[500:1000,10]/closestAchieveableRate[500:1000,10])
   #   nearm = mean(closestAchieveableRate/closestAchieveableRate)
   #   strongm = mean(oSINRAchieveableRate/closestAchieveableRate)
      
   #   achm = mean(handovers[500:length(achieveableRate)])*60/100
   #   nearm = mean(nearestHandovers)*60/100
  #    strongm = mean(oSINRHandovers)*60/100

  
      #save the performance values and the DQN weights
  save(handovers,closestHandovers,oSINRHandovers,current_state_replay,action_replay,reward_replay,next_state_replay,oSINRAchieveableRate,closestAchieveableRate,achieveableRate,file="REQIBAEvaluation.RData")
  model %>% save_model_weights_hdf5("REQIBAweights.h5")
    }
}  


grid(nx = NULL, ny = NULL, col = "darkgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)


#legend("bottomright",                       # x-y coordinates for location of the legend  
#       legend=c("DQN","SINR (omni)","Closest"),      # Legend labels  
#       col=c("black","red","blue"),   # Color of points or lines  
#       lty=c(1,1,1,1,1,1),                    # Line type  
#       lwd=c(8,8,8,8,8),                    # Line width  
#       #       pch=c(15,16,17),
#       cex=1.2
#) 

