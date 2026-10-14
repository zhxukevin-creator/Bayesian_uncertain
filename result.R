d1=read.csv("D:/wpi/phd research/paper with nandram and jnk rao/simdata_0.3.csv")
d2=read.csv("D:/wpi/phd research/paper with nandram and jnk rao/simdata_0.5.csv")
d3=read.csv("D:/wpi/phd research/paper with nandram and jnk rao/simdata_0.8.csv")
#when fit the model we only consider the survey weights of probability sample
propsenti=d2$x1+0.2*d2$x2+0.1*d2$x3-18.5

max(propsenti)/min(propsenti)
#之前设置的
b=d2$x1+0.2*d2$x2+0.1*d2$x3-85
propsenti_n=exp(b)/(1+exp(b))
sum(propsenti_n)
d1['propsenty_n']=propsenti_n
d2['propsenty_n']=propsenti_n
d3['propsenty_n']=propsenti_n
#让propenisty和y有关系
b=-0.01*d2$x1+0.5*d2$x2+0.5*d2$x3-2.75
propsenti_n=exp(b)/(1+exp(b))
sum(propsenti_n)
d1['propsenty_n']=propsenti_n
d2['propsenty_n']=propsenti_n
d3['propsenty_n']=propsenti_n
#d1['propsenty']=300*propsenti/sum(propsenti)
#d2['propsenty']=300*propsenti/sum(propsenti)
#d3['propsenty']=300*propsenti/sum(propsenti)


write.csv(d1,"D:/wpi/phd research/paper with nandram and jnk rao/simdata_0.3.csv",row.names = F)
write.csv(d2,"D:/wpi/phd research/paper with nandram and jnk rao/simdata_0.5.csv",row.names = F)
write.csv(d3,"D:/wpi/phd research/paper with nandram and jnk rao/simdata_0.8.csv",row.names = F)
summary(d1)
summary(lm(d1$y1~d1$x1+d1$x2+d1$x3))
######################################
#method1:unp_r
#method2:unnp_r
#method3:unp
#method4:unnp
#method5:tr_r
#method6:tr
#######################################
r_0.3=data.frame(matrix(nrow = 6,ncol = 6))
colnames(r_0.3)=c('PM','PSD','ARB','ARMSE','WIDTH','COV','BIAS^2','PSD')
r1=read.csv("D:/wpi/phd research/paper with nandram and jnk rao/result/sim_0.3/method_1_results.csv")
muy=27.24
mean(r1$posterior_mean)
mean(r1$posterior_std)
mean(abs(r1$posterior_mean-muy)/muy)
mean(sqrt((r1$posterior_mean-muy)^2+r1$posterior_std^2))
mean(r1$hpd_upper-r1$hpd_lower)
mean((r1$hpd_lower<muy&r1$hpd_upper>muy))
##############################################################
r_0.3 <- data.frame(matrix(nrow = 6, ncol = 9))
colnames(r_0.3) <- c('PM', 'PSD', 'AB','ARB', 'ARMSE', 'WIDTH', 'COV','BIAS^2','PSD')
rownames(r_0.3)=c('unp_r','unnp_r','unp','unnp','tr_r','tr')
muy <- 27.24
base_path <- "D:/wpi/phd research/paper with nandram and jnk rao/result/0.3/NEW_P/"

#for (i in 1:6) {
#  r <- read.csv(paste0(base_path, "method_", i, "_results.csv"))
#  
#  r_0.3[i, ] <- c(
#    mean(r$posterior_mean),
#    mean(r$posterior_std),
#   mean(abs(r$posterior_mean - muy) / muy),
#    mean(sqrt((r$posterior_mean - muy)^2 + r$posterior_std^2)),
#    mean(r$hpd_upper - r$hpd_lower),
#   mean(r$hpd_lower < muy & r$hpd_upper > muy)
# )
#}

for (i in 1:6) {
  # 读取两个 round 的文件
  file1 <- paste0(base_path, "method_", i, "_round5_results.csv")
  file2 <- paste0(base_path, "method_", i, "_round6_results.csv")
  file3=paste0(base_path, "method_", i, "_round7_results.csv")
  r1 <- read.csv(file1)
  r2 <- read.csv(file2)
  r3=read.csv(file3)
  # 将两个数据框按行合并
  r_combined <- rbind(r1, r2,r3)
  #file1=paste0(base_path, "method_", i, "_round5_results.csv")
  #r_combined=read.csv(file1)
  # 基于合并后的数据计算统计量
  r_0.3[i, ] <- c(
    mean(r_combined$posterior_mean),
    mean(r_combined$posterior_std),
    mean(abs(r_combined$posterior_mean - muy)),
    mean(abs(r_combined$posterior_mean - muy) / muy),
    mean(sqrt((r_combined$posterior_mean - muy)^2 + r_combined$posterior_std^2)),
    mean(r_combined$hpd_upper - r_combined$hpd_lower),
    mean(r_combined$hpd_lower < muy & r_combined$hpd_upper > muy),
    mean((r_combined$posterior_mean - muy)^2),
    mean( r_combined$posterior_std)
  )
}
#############################################################
r_0.5 <- data.frame(matrix(nrow = 6, ncol = 9))
colnames(r_0.5) <- c('PM', 'PSD', 'AB','ARB', 'ARMSE', 'WIDTH', 'COV','BIAS^2','PSD')
rownames(r_0.5)=c('unp_r','unnp_r','unp','unnp','tr_r','tr')

muy <- 27.25
base_path <- "D:/wpi/phd research/paper with nandram and jnk rao/result/0.5/NEW_P/"

for (i in 1:6) {
  # 读取两个 round 的文件
  file1 <- paste0(base_path, "method_", i, "_round5_results.csv")
  file2 <- paste0(base_path, "method_", i, "_round6_results.csv")
  file3=paste0(base_path, "method_", i, "_round7_results.csv")
  r1 <- read.csv(file1)
  r2 <- read.csv(file2)
  r3=read.csv(file3)
  # 将两个数据框按行合并
  r_combined <- rbind(r1, r2,r3)
  
  #file1=paste0(base_path, "method_", i, "_round5_results.csv")
  #r_combined=read.csv(file1)
  # 基于合并后的数据计算统计量
  r_0.5[i, ] <- c(
    mean(r_combined$posterior_mean),
    mean(r_combined$posterior_std),
    mean(abs(r_combined$posterior_mean - muy)),
    mean(abs(r_combined$posterior_mean - muy) / muy),
    mean(sqrt((r_combined$posterior_mean - muy)^2 + r_combined$posterior_std^2)),
    mean(r_combined$hpd_upper - r_combined$hpd_lower),
    mean(r_combined$hpd_lower < muy & r_combined$hpd_upper > muy),
    mean((r_combined$posterior_mean - muy)^2),
    mean( r_combined$posterior_std)
  )
}
########################################################
r_0.8 <- data.frame(matrix(nrow = 6, ncol = 9))
colnames(r_0.8) <- c('PM', 'PSD', 'AB','ARB', 'ARMSE', 'WIDTH', 'COV','BIAS^2','PSD')
rownames(r_0.8)=c('unp_r','unnp_r','unp','unnp','tr_r','tr')

muy <- 27.25
base_path <- "D:/wpi/phd research/paper with nandram and jnk rao/result/0.8/NEW_P/"

for (i in 1:6) {
  # 读取两个 round 的文件
  file1 <- paste0(base_path, "method_", i, "_round5_results.csv")
  file2 <- paste0(base_path, "method_", i, "_round6_results.csv")
  file3=paste0(base_path, "method_", i, "_round7_results.csv")
  r1 <- read.csv(file1)
  r2 <- read.csv(file2)
  r3=read.csv(file3)
  # 将两个数据框按行合并
  r_combined <- rbind(r1, r2,r3)
  
  #file1=paste0(base_path, "method_", i, "_round5_results.csv")
  #r_combined=read.csv(file1)
  # 基于合并后的数据计算统计量
  r_0.8[i, ] <- c(
    mean(r_combined$posterior_mean),
    mean(r_combined$posterior_std),
    mean(abs(r_combined$posterior_mean - muy)),
    mean(abs(r_combined$posterior_mean - muy) / muy),
    mean(sqrt((r_combined$posterior_mean - muy)^2 + r_combined$posterior_std^2)),
    mean(r_combined$hpd_upper - r_combined$hpd_lower),
    mean(r_combined$hpd_lower < muy & r_combined$hpd_upper > muy),
    mean((r_combined$posterior_mean - muy)^2),
    mean( r_combined$posterior_std)
  )
}


write.csv(r_0.3,"D:/wpi/phd research/paper with nandram and jnk rao//result/0.3.csv",row.names = F)
write.csv(r_0.5,"D:/wpi/phd research/paper with nandram and jnk rao/result/0.5.csv",row.names = F)
write.csv(r_0.8,"D:/wpi/phd research/paper with nandram and jnk rao/result/0.8.csv",row.names = F)

data=read.csv("D:/wpi/phd research/2022 fall/non-prop-combine/bal-rao.txt",sep="")
# names(data)=c("nothing","county","age","nothin2","sex","bmi","race","weights")
names(data)=c("nothing","county","age","race","sex","bmi","nothing2","weights")

sum(data$bmi*data$weights)/sum(data$weights)

pro_data=data[data$county==1| data$county==19|data$county==111,]
sum(pro_data$bmi*pro_data$weights)/sum(pro_data$weights)
