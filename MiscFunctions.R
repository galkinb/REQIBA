
##Get the achieveable rate, when the base stations have antennas that use Ramy Amer's directional model
getDualRateRamyAntenna = function(x,BHBS,BSh,withFading,LOS,BHtilt,iBSBH,iBSBHHeight,al,an,mal,man,Nt,PWRgain,N,alpha,beta,gamma){
  dist = sqrt((BHBS[1]-x[1])^2+(BHBS[2]-x[2])^2+(BSh-x[3])^2)
  
  rdist = sqrt((BHBS[1]-x[1])^2+(BHBS[2]-x[2])^2)
  angle = (atan2(x[3]-BSh,rdist))
  hangle = atan2((BHBS[2]-x[2]),(BHBS[1]-x[1]))
  g=(1/Nt)*((sin(Nt*pi*(sin(angle))/2)^2)/(sin(pi*(sin(angle))/2)^2))

  g=g*PWRgain
  
  
  if(LOS==TRUE){
    PL = dist^(-al)
    if(withFading==TRUE){
      PL = PL*rgamma(n=1,rate=mal,shape=mal)
    }
  }
  else{
    PL = dist^(-an)
    if(withFading==TRUE){
      PL = PL*rgamma(n=1,rate=man,shape=man) 
    }
  }
  
  I=0
  
  if(length(iBSBH[,1])>0){
    for(i in 1:length(iBSBH[,1])){
      idist = sqrt((iBSBH[i,1]-x[1])^2+(iBSBH[i,2]-x[2])^2+(BSh-x[3])^2)
      
      rdist =sqrt((iBSBH[i,1]-x[1])^2+(iBSBH[i,2]-x[2])^2)
      angle = (atan2(x[3]-BSh,rdist))
      hangle = atan2((iBSBH[i,2]-x[2]),(iBSBH[i,1]-x[1]))
      ig=(1/Nt)*((sin(Nt*pi*(sin(angle))/2)^2)/(sin(pi*(sin(angle))/2)^2))
      ig=ig*PWRgain
      
      
      test = (iBSBHHeight[i]<=x[3])
      if(test==TRUE){
        iLOS = TRUE
        iPL = idist^(-al) 
        if(withFading==TRUE){
          iPL =iPL*rgamma(n=1,rate=mal,shape=mal)
        }
      }
      else{
        iLOS = FALSE
        iPL = idist^(-an) 
        if(withFading==TRUE){
          iPL =iPL*rgamma(n=1,rate=man,shape=man)
        }
      }
      I = I+ig*iPL
    } 
  }
  
  
  SINR = (g*PL/(N+I))
  return(log2(1+SINR))
  
} 

#get the interfering BSs which fall inside the illuminated area of the UAV directional antenna when it is pointed at a serving BS
getInt2 = function(x,int,BHBS,grid,UAVBHBW){
  intBSx = grid$x[int]-x[1]
  intBSy = grid$y[int]-x[2]
  BSx = BHBS[1]-x[1]
  BSy = BHBS[2]-x[2]
  
  di = acos((BSx*intBSx+BSy*intBSy)/(sqrt(BSx^2+BSy^2)*sqrt(intBSx^2+intBSy^2)))
  if(length(find(is.na(di)))>0){
    di[is.na(di)] = Inf
  }
  int = int[abs(di)<UAVBHBW/2]
  return(int)
} 

#statistical LOS probability
Plos = function(htx,hrx,r,alpha,beta,gamma){
  n = floor((r/1000)*sqrt(alpha*beta))
  a = 0:max(0,n-1)
  P = prod(1-exp(-((htx-(a+1/2)*(htx-hrx)/(n))^2)/(2*gamma^2)))
  if(n==0 && is.na(P))
  {P = 1}
  return(P)
}
