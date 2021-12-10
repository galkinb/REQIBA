##Given a certain BS to connect to and a certain starting height, have the UAV pick the best height and BS association as it travels in a straight line
rm(list=ls())
library(spatstat)
library(VGAM)
library(hypergeo)
library(keras)
library(parallel)
source("MiscFunctions.R")


#number of MC trials, episodes and steps
MCtrials = 1000
episodes = 1
steps = 100

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
maxH = 200

velocity = 10

#Base station height
BSh = 30
windowWidth = 5000


#Base station density
BHdensity = 5/(1000^2)

UAVBHBW = pi*1/4

Tb = -6
Tu = 0

al = 2.1
an = 4
N=10^(-9)
mal = 10
man = 1
BStx = 40


Freq = 2*10^9

RBbeamwidth = 180000


N = -174+10*log10(20*10^6)+10
N = 10^(N/10)/1000

BStilt=-10
BHtilt=-10

cores=6


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


h = seq(from=20,to=200,by=180/10)

#We compare our REQIBA algorithm to a random walk, angle-based, minimum average distance, and the IPNN used in isolation
#Our KPIs are the average episode-wide throughput and the number of handovers per min
loadAchieveableRate = 0#zeros(nrow=steps,ncol=length(Tu))
mdistAchieaveableRate = 0
angleAchieaveableRate = 0
intAchieveableRate = 0
nearestAchieveableRate = 0

loadA = vector(length=length(h))
mdistA = vector(length=length(h))
angleA = vector(length=length(h))
intA = vector(length=length(h))
nearestA = vector(length=length(h))

loadHandovers = 0
mdistHandovers = 0
angleHandovers = 0
intHandovers = 0
nearestHandovers = 0

loadH = vector(length=length(h))
mdistH = vector(length=length(h))
angleH = vector(length=length(h))
intH = vector(length=length(h))
nearestH = vector(length=length(h))


##UAV antenna gain
UAVBHgain = 4*pi/((UAVBHBW/2)^2)
#near-field pathloss
K = (((3*10^8)/Freq)/(4*pi))^2

#load the evaluation results of the REQIBA solution, so we can plot our heuristic results side-by-side
load("REQIBAEvaluation.RData")

#multiple MC trials. In each trial we generate the environment and then run the algorithm over multiple episodes
for(l in 1:length(h)){
for(m in 1:MCtrials){
    print(m)

    cPm= vector(length=MCtrials)
    cPmn = vector(length=MCtrials)
    cPmML= vector(length=MCtrials)
    
    whichNthBS = vector(length=MCtrials)
    
    
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
      BSload[] = 1
      
      ##which network a given BS belongs to
      ##Note, I wrote this code for multi-network scenarios where the UAV can choose from several different operator networks
      ##We decided against looking at multi-network scenarios for the TVT paper
      ##Nonetheless the dataset and the resulting neural network is capable of distinguishing between different networks, we simply don't use that function in the simulations
      whichNetwork=ones(nrow=BHppp$n,ncol=1)#floor(runif(n=BHppp$n,min=1,max=4))
      
    #  h =h[l]#seq(from=minH-10,to=maxH+10,by=1)
      
      #generate the state spaces for all of the steps in the MC trial
      #the iteration function generates all of the state observations a priori, as well as the rewards for each of the possible actions
      #then we have the main loop below which goes through the states, takes actions, and gets rewards
      #we separate out the state generation and the decision loop because it seems to be more computationally efficient to call this loop in one go and generate the entire episode worth of data, using parallel threads
      iteration = function(q){
        j = floor(q/steps)+1
        w = q%%steps+1

        LOS = vector(length=BHppp$n)
        
        measuredpower = vector(length=BHppp$n)

        connectedto=floor(runif(n=1,min=1,max=(BScand+1))) #the UAV is connected to one of the BScand closest BSs (so it then decides whether to stay connected or change)
        numInterferers = zeros(nrow=BScand,ncol=BScandi)
        intLOS = zeros(nrow=BScand,ncol=BScandi)
        intP = zeros(nrow=BScand,ncol=BScandi)
        distances = vector(length=BScand)
        meanDistances = vector(length=BScand)
        cov = vector(length=BScand)
        angleVector = vector(length=BHppp$n)
        
        
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
          z = 1:steps
          mdist[i] = mean(sqrt((BHppp$x[i]-uavx[z])^2+(BHppp$y[i])^2))/1000
          g=(1/Nt)*((sin(Nt*pi*(sin(angle))/2)^2)/(sin(pi*(sin(angle))/2)^2))
          g = g*BStx*K
          angleVector[i] = acos((1*(BHppp$x[i]-uavx[w])+0*BHppp$y[i])/(sqrt(1^2+0^2)*sqrt((BHppp$x[i]-uavx[w])^2+BHppp$y[i]^2)))
          if(LOS[i]==TRUE){
            measuredpower[i]=g*(sqrt((BHppp$x[i]-uavx[w])^2+(BHppp$y[i])^2+(BSh-h[l])^2))^(-al)  
          }else{measuredpower[i]=g*(sqrt((BHppp$x[i]-uavx[w])^2+(BHppp$y[i])^2+(BSh-h[l])^2))^(-an)}
        }
        
        #now get the SINR for each BS
        oSINR = vector(length=BHppp$n)
        for(i in 1:BHppp$n){
          foo = 1:BHppp$n
          foo = foo[foo!=i]
          oSINR[i] = measuredpower[i]/(sum(measuredpower[foo])+N)
        }
        
        
        iBSBHHeight = vector(length=BHppp$n)
        iBSBHHeight[LOS==TRUE]=0
        iBSBHHeight[LOS==FALSE]=Inf
        order1 =order(load,decreasing=TRUE)
        order2 =order(mdist,decreasing=FALSE)
        order3 =order(angleVector,decreasing=FALSE)
        order4 = order(rdist,decreasing=FALSE)
        order = c(order1[1],order2[1],order3[1],order4[1:10])
        
        
        achieveableRate = vector(length=BScand)
        SINR = vector(length=BScand)
        RBsneeded = vector(length=BScand)
        
        
        ##get the rate that would be achieved from each BS through the directional antenna
        for(i in 1:BScand){
          BHdist = sqrt((BHppp$x-uavx[w])^2+(BHppp$y)^2)
          BHBS = c(BHppp$x[order[i]],BHppp$y[order[i]])
          
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
       #   RBsneeded[i] = 0#ceil(minCCdatarate/(200000*specEff)) #not using this
          SINR[i]=2^(specEff)-1 #SINR during the timestep, with the multipath fading effects ignored due to the timestep duration
          specEff = getDualRateRamyAntenna(x=c(uavx[w],0,hopt),BHBS=BHBS,BSh=BSh,withFading=TRUE,LOS=LOS[ind],iBSBH=cbind(BHint[,1],BHint[,2]),BHtilt=BHtilt,iBSBHHeight=BHH,Nt=Nt,al=al,an=an,mal=mal,man=man,PWRgain=BStx*UAVBHgain*K,N=N,alpha=alpha,beta=beta,gamma=gamma)
      #    RBsneeded[i] = 0#ceil(minCCdatarate/(200000*specEff)) #not using this
          cov[i]=2^(specEff)-1 #instantaneous SINR value, including the multipath fading effects (We would use this if we were interested in the coverage probability)
        }
        
        #sort the BSs in terms of lowest load (unused), shortest mean distance, best aligned angle, or shortest distance
        order1 =order(load,decreasing=TRUE)
        order2 =order(mdist,decreasing=FALSE)
        order3 =order(angleVector,decreasing=FALSE)
        order4 = order(rdist,decreasing=FALSE)
        order = c(order1[1],order2[1],order3[1],order4[1:10])
        
        foo = 1:BScand
        
        return(c(as.vector(t(numInterferers)),as.vector(t(intLOS)),load[order[foo]],SINR[foo],order[foo]))
      }
      
      #generate the environmental observations across all of the steps and for each possible UAV-BS association, using multi-threaded process
      X=0:(steps-1)
      opt = mclapply(X=X,FUN=iteration,mc.cores=cores)
      
      Int = (1):(BScand*BScandi)
      IntLOS = (Int[BScand*BScandi]+1):(Int[BScand*BScandi]+BScand*BScandi)
      load = (IntLOS[BScand*BScandi]+1):(IntLOS[BScand*BScandi]+BScand)
      sSINR = (load[BScand]+1):(load[BScand]+BScand)
      BSID = (sSINR[BScand]+1):(sSINR[BScand]+BScand)
      
      mcInt =array(dim=c(1,steps,BScand*BScandi))
      mcSINR =array(dim=c(1,steps,BScand))
      mcBSID =array(dim=c(1,steps,BScand))
      mcLoad =array(dim=c(1,steps,BScand))
      mcIntLOS =array(dim=c(1,steps,BScand*BScandi))
      
      #extract and store the data we generated in the iteration loop
      for(k in 0:(steps-1)){
        j = floor(k/steps)+1
        w = k%%steps+1
        mcInt[j,w,]=opt[[k+1]][Int]
        mcSINR[j,w,] =opt[[k+1]][sSINR]
        mcBSID[j,w,]=opt[[k+1]][BSID]
        mcLoad[j,w,]=opt[[k+1]][load]
        mcIntLOS[j,w,]=opt[[k+1]][IntLOS]
      }
      
      #apply the IPNN model to predict interference powers from the interference distances, channel types and UAV heights for all steps in the episode
      foo = as.vector(t(mcInt[1,,]))/1000
      fooL = as.vector(t(mcIntLOS[1,,]))
      fooH = vector(length=length(fooL))
      fooH[] = h[l]/300
      check=which(foo==0)
      reg = regressionModel %>% predict(cbind(foo,fooL,fooH))
      g=10^(reg)
      g[check]=0
      rInt = matrix(g,nrow=steps,ncol=BScand*BScandi,byrow=TRUE)
    

      #for each episode in a MC trial (by default, there is only 1 episode per MC trial, we can do several episodes per MC trial if we want to have the UAV travel the same path through the same environment multiple times)
      for(k in ((m-1)*episodes+1):(m*episodes)){
      #starting height
      UAVh = h[l]#floor(runif(n=1,min=minH,max=maxH))
      whichBS = floor(runif(n=1,min=1,max=(BScand+1)))
      UAVass = mcBSID[1,1,whichBS]
      meanisHandover = FALSE
      loadisHandover=FALSE
      angleisHandover=FALSE
      intisHandover=FALSE
      nearestisHandover=FALSE
      meanUAVh = h[l]
      meanUAVass = mcBSID[1,1,2]
      loadUAVh = h[l]
      loadUAVass = mcBSID[1,1,1]
      angleUAVh = h[l]
      angleUAVass = mcBSID[1,1,3]
      intUAVh = h[l]
      intUAVass = 0
      nearestUAVh = h[l]
      nearestUAVass = mcBSID[1,1,4]
      cat("MC Trial: ",m," Episode: ",k,"\n")
      load_episode_reward = 0
      load_episode_handovers = 0
      mean_episode_reward = 0
      mean_episode_handovers = 0
      int_episode_reward = 0
      int_episode_handovers = 0
      angle_episode_reward = 0
      angle_episode_handovers = 0
      nearest_episode_reward = 0
      nearest_episode_handovers = 0
      
      for(j in 1:(steps-1)){
    #    print(j)
        foo = 1
        
        #UNUSED
        loadaction= 1
        if(mcBSID[1,j,loadaction]!=loadUAVass){
        loadisHandover=TRUE
        load_episode_handovers=load_episode_handovers+1
        }
        loadUAVass = mcBSID[1,j,loadaction]
        
        #connect to BS with shortest mean distance
        meanaction= 2
        if(mcBSID[1,j,meanaction]!=meanUAVass){
          meanisHandover=TRUE
          mean_episode_handovers=mean_episode_handovers+1
        }
        meanUAVass = mcBSID[1,j,meanaction]
        
        #connect to BS with best aligned angle
        angleaction= 3
        if(mcBSID[1,j,angleaction]!=angleUAVass){
          angleisHandover=TRUE
          angle_episode_handovers=angle_episode_handovers+1
        }
        angleUAVass = mcBSID[1,j,angleaction]
        
        #connect to closest BS
        nearestaction= 4
        if(mcBSID[1,j,nearestaction]!=nearestUAVass){
          nearestisHandover=TRUE
          nearest_episode_handovers=nearest_episode_handovers+1
        }
        nearestUAVass = mcBSID[1,j,nearestaction]
        
        #connect to BS with smallest interference, as per IPNN
        rrInt = vector(length=BScand)
        for(i in 1:BScand){
          rrInt[i] = sum(rInt[j,((i-1)*BScandi+1):(i*BScandi)])
        }
        rrInt[1:3]=Inf
        
        intaction= which(rrInt==min(rrInt))
        if(length(intaction)>1){
        intaction = min(intaction)  
        }
        if(mcBSID[1,j,intaction]!=intUAVass){
          if(j!=1){
          intisHandover=TRUE
          int_episode_handovers=int_episode_handovers+1
          }
        }
        intUAVass = mcBSID[1,j,intaction]
        
        #calculate the throughputs
        if(loadisHandover==FALSE){
        loadreward = RBbeamwidth*mcLoad[foo,j,loadaction]*log2(1+mcSINR[foo,j,loadaction])/(10^6)
        }
        else{loadreward = handoverpenalty*RBbeamwidth*mcLoad[foo,j,loadaction]*log2(1+mcSINR[foo,j,loadaction])/(10^6)}
        
        if(meanisHandover==FALSE){
          meanreward = RBbeamwidth*mcLoad[foo,j,meanaction]*log2(1+mcSINR[foo,j,meanaction])/(10^6)
        }
        else{meanreward = handoverpenalty*RBbeamwidth*mcLoad[foo,j,meanaction]*log2(1+mcSINR[foo,j,meanaction])/(10^6)}
        
        if(angleisHandover==FALSE){
          anglereward = RBbeamwidth*mcLoad[foo,j,angleaction]*log2(1+mcSINR[foo,j,angleaction])/(10^6)
        }
        else{anglereward = handoverpenalty*RBbeamwidth*mcLoad[foo,j,angleaction]*log2(1+mcSINR[foo,j,angleaction])/(10^6)}
        
        if(nearestisHandover==FALSE){
          nearestreward = RBbeamwidth*mcLoad[foo,j,nearestaction]*log2(1+mcSINR[foo,j,nearestaction])/(10^6)
        }
            else{nearestreward = handoverpenalty*RBbeamwidth*mcLoad[foo,j,nearestaction]*log2(1+mcSINR[foo,j,nearestaction])/(10^6)}
        
        if(intisHandover==FALSE){
          intreward = RBbeamwidth*mcLoad[foo,j,intaction]*log2(1+mcSINR[foo,j,intaction])/(10^6)
        }
        else{intreward = handoverpenalty*RBbeamwidth*mcLoad[foo,j,intaction]*log2(1+mcSINR[foo,j,intaction])/(10^6)}
        
      
        loadisHandover=FALSE
        meanisHandover=FALSE
        angleisHandover=FALSE
        intisHandover = FALSE
        nearestisHandover=FALSE
        
        #update cumulative rewards
        load_episode_reward=loadreward+load_episode_reward
        mean_episode_reward=meanreward+mean_episode_reward
        angle_episode_reward = anglereward+ angle_episode_reward 
        int_episode_reward = intreward+ int_episode_reward 
        nearest_episode_reward = nearestreward+ nearest_episode_reward 
        
        
        #store the rolling averages
      }
      
      if(k==1){
        loadAchieveableRate = c(load_episode_reward)
        meanAchieveableRate = c(mean_episode_reward)
        angleAchieveableRate = c(angle_episode_reward)
        intAchieveableRate = c(int_episode_reward)
        nearestAchieveableRate = c(nearest_episode_reward)
        loadHandovers=c(load_episode_handovers)
        meanHandovers= c(mean_episode_handovers)
        angleHandovers = c(angle_episode_handovers)  
        intHandovers = c(int_episode_handovers)  
        nearestHandovers = c(nearest_episode_handovers)  
      }
      else{
        loadAchieveableRate = c(loadAchieveableRate,load_episode_reward)
        meanAchieveableRate = c(meanAchieveableRate,mean_episode_reward)
        angleAchieveableRate = c(angleAchieveableRate,angle_episode_reward)
        intAchieveableRate = c(intAchieveableRate,int_episode_reward)
        nearestAchieveableRate = c(nearestAchieveableRate,nearest_episode_reward)
        loadHandovers=c(loadHandovers,load_episode_handovers)
        meanHandovers= c(meanHandovers,mean_episode_handovers)
        angleHandovers = c(angleHandovers,angle_episode_handovers)  
        intHandovers = c(intHandovers,int_episode_handovers) 
        nearestHandovers = c(nearestHandovers,nearest_episode_handovers)
      }
      
      
   #   plot(x=1:(k),y=loadAchieveableRate/nearestAchieveableRate,col='red',type='l',xlab="Episodes",ylab="Normalised Throughput",ylim=c(0,3))
     #     lines(x=1:k,y=meanAchieveableRate/nearestAchieveableRate,col='green')
  #    lines(x=1:(k),y=nearestAchieveableRate/nearestAchieveableRate,col='blue')
  #    lines(x=1:(k),y=intAchieveableRate/nearestAchieveableRate)
  #    lines(x=1:(k),y=angleAchieveableRate/nearestAchieveableRate,col='purple')
      
      
   #   avg = vector(length=k)
    #    for(i in 50:k){
    #    avg[i] = mean(achieveableRate[(i-50):i]/closestAchieveableRate[(i-50):i])  
    #    }
      
      }
      
  
}
  
  #get normalised mean throughputs and plot (comment out code as needed)
  loadA[l] = mean(loadAchieveableRate/nearestAchieveableRate)
  mdistA[l] = mean(meanAchieveableRate/nearestAchieveableRate)
  angleA[l] = mean(angleAchieveableRate/nearestAchieveableRate)
  intA[l] = mean(intAchieveableRate/nearestAchieveableRate)
  nearestA[l] = mean(nearestAchieveableRate/nearestAchieveableRate)
  
       plot(x=h,y=mdistA,col='orange',type='l',xlab="UAV height (m)",ylab="Normalised Throughput",ylim=c(0,2),lwd=3,cex.lab=1.3,cex.axis=1.3)
        lines(x=h,y=nearestA,col='blue',lwd=3,)
        lines(x=h,y=intA,col='green',lwd=3,)
        lines(x=h,y=angleA,col='purple',lwd=3,)
        points(x=h,y=mdistA,col='orange',pch=4,cex=2)
        points(x=h,y=nearestA,col='blue',pch=15,cex=2)
        points(x=h,y=intA,col='green',pch=16,cex=2)
        points(x=h,y=angleA,col='purple',pch=17,cex=2)
  
  #get mean handovers and plot
  loadH[l] =mean(loadHandovers)
  mdistH[l] = mean(meanHandovers)
  angleH[l] = mean(angleHandovers)
  intH[l] = mean(intHandovers)
  nearestH[l] = mean(nearestHandovers)
  
  plot(x=h,y=mdistH*60/100,col='orange',type='l',xlab="UAV height (m)",ylab="Handovers per Min.",ylim=c(0,25),lwd=3,cex.lab=1.3,cex.axis=1.3)
#  lines(x=h,y=mdistH*60/100,col='orange')
  lines(x=h,y=nearestH*60/100,col='blue',lwd=3)
  lines(x=h,y=intH*60/100,col='green',lwd=3)
  lines(x=h,y=angleH*60/100,col='purple',lwd=3)
  points(x=h,y=mdistH*60/100,col='orange',pch=4,cex=2)
  #      lines(x=h,y=mdistA,col='orange')
  points(x=h,y=nearestH*60/100,col='blue',pch=15,cex=2)
  points(x=h,y=intH*60/100,col='green',pch=16,cex=2)
  points(x=h,y=angleH*60/100,col='purple',pch=17,cex=2)
  
  #process the evaluation data from the REQIBA script
  dqnA = vector(length=length(h))
  dqnH = vector(length=length(h))
  strongestA = vector(length=length(h))
  strongestH = vector(length=length(h))
  for(i in 1:length(h)){
  dqnA[i] = mean(achieveableRate[500:1000,i]/closestAchieveableRate[500:1000,i])
  dqnH[i] = mean(handovers[500:1000,i])
  strongestA[i]=mean(oSINRAchieveableRate[,i]/closestAchieveableRate[,i])
  strongestH[i]=mean(oSINRHandovers[,i]) 
  }
  
  lines(x=h,y=dqnA,lwd=3)
  lines(x=h,y=strongestA,col='red',lwd=3)
  points(x=h,y=dqnA,pch=8,lwd=3,cex=2)
  points(x=h,y=strongestA,col='red',pch=18,cex=2)
  
  lines(x=h,y=dqnH*60/100,lwd=3)
  lines(x=h,y=strongestH*60/100,col='red',lwd=3)
  points(x=h,y=dqnH*60/100,lwd=3,pch=8,cex=2)
  points(x=h,y=strongestH*60/100,col='red',pch=18,cex=2)
  
  
  save(loadA,mdistA,nearestA,intA,angleA,loadH,mdistH,angleH,intH,nearestH,file="Heuristics.RData")
    }  



grid(nx = NULL, ny = NULL, col = "darkgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)


legend("bottomright",                       # x-y coordinates for location of the legend  
       legend=c("Closest","IPNN+DDQN","IPNN","SINR (omni)","Mean Distance","Angle"),      # Legend labels  
       col=c("blue","black","green","red","orange","purple"),   # Color of points or lines
       #   horiz=TRUE,
       lty=c(1,1,1,1,1,1),                    # Line type  
       lwd=c(3,3,3,3,3,3),                    # Line width  
       pch=c(15,8,16,18,4,17),
       cex=1.2,
       #     text.width=c(1,0,0.7,1,1,1),
) 



#legend("topright",                       # x-y coordinates for location of the legend  
#       legend=c("Baseline","ANN single network","ANN 3 network intra-handover","ANN 3 network inter-handover"),      # Legend labels  
#       col=c("black","red","blue","green"),   # Color of points or lines  
 #      lty=c(1,1,1,1,1),                    # Line type  
#       lwd=c(8,8,8),                    # Line width  
#       pch=c(15,16,17),
#       cex=1.2
#) 
