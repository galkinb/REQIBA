##generate the dataset for the IPNN training.
#We simulate the UAV flying into an urban environment and aligning its directional antenna with a randomly chosen BS
#It records the distance to the BS (known from map), it's height above ground, the LOS channel type (known from topological map)
#and the resulting received signal power

rm(list=ls())
library(spatstat)
library(hypergeo)
library(VGAM)
library(parallel)
source("MiscFunctions.R")

MCtrials = 1000000

#building params
alpha = 0.5
beta = 300
gamma = 20

#building parameters
buildDens = 300/(1000^2)
buildWidth = 40
buildR = buildWidth/(2*sin(pi/4))
heightParam = 20

rdist = 300
etilt = 10

#width of the simulation window, density of BSs
windowWidth = 5000
BHdensity = 5/(1000^2)


#number of antenna elements in the BS antennas
Nt= 8

#coverage probability threshold
Tu = 0

#LOS/NLOS pathloss exponents, LOS/NLOS fading parameter, BS transmit power (Watts), 
al = 2.1
an = 4
mal = 10
man = 1
BStx = 40

##Transmit frequency (for near-field pathloss calculation)
Freq = 2*10^9

#Noise power
N = -174+10*log10(20*10^6)+10
N = 10^(N/10)/1000

#MC cores
cores =12

##near-field pathloss parameter
K = (((3*10^8)/Freq)/(4*pi))^2

#Raytracing function for checking LOS (I don't currently use it for this code, it slows the calculations down compared to the other LOS function)
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

##dataset generation
iteration = function(m){
 if(m%%10000==0){
    print(m)
 }
  
  #randomly generate the UAV beamwidth
  UAVBHBW = pi/4#pi*1/runif(n=1,min=1,max=6)
  UAVBHgain = 4*pi/((UAVBHBW/2)^2)
  
  #randomise the the UAV height
  h = runif(n=1,min=0,max=300)
  
  BSh=30

  ##window in which we simulate BS distribution
  sWindow = owin(xrange=c(-windowWidth/2,windowWidth/2),yrange=c(-windowWidth/2,windowWidth/2))
  gen=FALSE
  while(gen==FALSE){
    BHppp = rpoispp(lambda=BHdensity,win=sWindow)
    if(BHppp$n>5){
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
  
  ##which network a given BS belongs to
  ##Note, I wrote this code for multi-network scenarios where the UAV can choose from several different operator networks
  ##I've decided to drop that in the paper we're writing as it gives us nothing of value. In the paper it's just one network
  ##Nonetheless the dataset and the resulting neural network is capable of distinguishing between different networks, we simply don't use that function in the simulations
  whichNetwork=ones(nrow=BHppp$n,ncol=1)#floor(runif(n=BHppp$n,min=1,max=4))
  
  LOS = 0

  measuredpower = 0
  
  #pick one of the BSs at random
  whichBS = floor(runif(n=1,min=1,max=(length(BHppp$n)+1)))
  
  #get the distance, channel type (LOS/NLOS) and the signal power received by the aligned directional antenna from that BS
    foo = runif(n=1,min=0,max=1)
    if(isLOS(buildings=buildings,buildR=buildR,buildH=buildH,x0=0,y0=0,x=BHppp$x[whichBS],y=BHppp$y[whichBS],h=h,BSh=BSh)){
      LOS=TRUE
    }
    else{LOS=FALSE
    }
    
    angle = (atan2(BSh-h,sqrt((BHppp$x[whichBS])^2+(BHppp$y[whichBS])^2)))
    rdist = (sqrt((BHppp$x[whichBS])^2+(BHppp$y[whichBS])^2))
    g=(1/Nt)*((sin(Nt*pi*(sin(angle))/2)^2)/(sin(pi*(sin(angle))/2)^2))
    
    g = g*BStx*K
    if(LOS==TRUE){
      measuredpower=g*(sqrt((BHppp$x[whichBS])^2+(BHppp$y[whichBS])^2+(BSh-h)^2))^(-al)  
    }else{measuredpower=g*(sqrt((BHppp$x[whichBS])^2+(BHppp$y[whichBS])^2+(BSh-h)^2))^(-an)}
  
  
  return(c(rdist,LOS,h,measuredpower))
}

X=1:MCtrials
opt = mclapply(X=X,FUN=iteration,mc.cores=cores)

  results=zeros(nrow=MCtrials,4)
#  
  for(k in 1:MCtrials){
    results[k,] = opt[[k]]

  }
save(results,file="IPNNDataset.RData",version=2)

