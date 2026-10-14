#need to generate N=20000 x1,x2,x3
#E(y)=27-0.01x1+0.5x2+0.5x3
#10% outlier so error~0.9N(0,sigma^2)+0.1N(0,25sigma^2),rho=0.3,0.5,0.8
set.seed(1)
N=20000
x1=runif(20000,min=20,max=90)
b=(27-0.01*x1)^0.1
prob=exp(b)/(1+exp(b))
x2=rbinom(20000,1,prob=prob)
b=(27-0.01*x1+0.5*x2)^0.1
prob=exp(b)/(1+exp(b))
x3=rbinom(20000,1,prob=prob)
mu = 27 - 0.01*x1 + 0.5*x2 + 0.5*x3
generate_error_outlier = function(N, sigma, outlier_rate = 0.1){
  
  outlier = rbinom(N,1,outlier_rate)
  
  error = ifelse(outlier==1,
                 rnorm(N,0,25*sigma),   # outlier part
                 rnorm(N,0,sigma))     # normal part
  
  return(error)
}

find_sigma = function(target_cor, mu){
  
  N = length(mu)
  
  f = function(sigma){
    
    error = generate_error_outlier(N, sigma)
    
    y = mu + error
    
    cor(y,mu) - target_cor
  }
  
  sigma = uniroot(f, c(0.01,10))$root
  
  return(sigma)
}

target_cor = c(0.3,0.5,0.8)

sigma1 = find_sigma(target_cor[1], mu)
sigma2 = find_sigma(target_cor[2], mu)
sigma3 = find_sigma(target_cor[3], mu)

error1 = generate_error_outlier(N, sigma1)
error2 = generate_error_outlier(N, sigma2)
error3 = generate_error_outlier(N, sigma3)

y1 = mu + error1
y2 = mu + error2
y3 = mu + error3

cor(y1,mu)
cor(y2,mu)
cor(y3,mu)


error1_clean = rnorm(N,0,sigma1)
error2_clean = rnorm(N,0,sigma2)
error3_clean = rnorm(N,0,sigma3)

# clean population outcomes
y1_clean = mu + error1_clean
y2_clean = mu + error2_clean
y3_clean = mu + error3_clean



simdata_0.3=data.frame(y1,x1,x2,x3)
write.csv(simdata_0.3,"simdata_0.3.csv",row.names = FALSE)
simdata_0.5=data.frame(y2,x1,x2,x3)
write.csv(simdata_0.5,"simdata_0.5.csv",row.names = FALSE)
simdata_0.8=data.frame(y3,x1,x2,x3)
write.csv(simdata_0.8,"simdata_0.8.csv",row.names = FALSE)

