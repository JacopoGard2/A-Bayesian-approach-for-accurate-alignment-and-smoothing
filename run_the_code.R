##########################
# Libraries 
##########################
library(splines)
library(MASS)
library(fda)
library(ddalpha)
library(emdbook)
library(RcppArmadillo)
library(splines2)
library(mvnfast)
library(coda)
library(LearnBayes)
library(lattice)
library(mvtnorm)
library(clue)
##########################
# Model hyper-parameters 
##########################
n_knots_beta = 10
n_knots_gamma = 3
n_knots_phi = 4
q = n_knots_phi + 4

degree_beta = 3
degree_gamma = 3
degree_phi = 3
intercept_beta = T
intercept_gamma = T
intercept_phi = T

a_eps = 3000
b_eps = 5000
a_lambda = 1200
b_lambda = 3500
a_gamma = 1000
b_gamma = 2000

c_star = 20
phi_star = seq(0,1,len=q)
a_f = rep(0,length(phi_star)-1)
a_f[1] = c_star * phi_star[2]
for(j in 2:(q-2)){
  a_f[j] = c_star*(phi_star[j+1] - phi_star[j])
}
a_f[q-1] = c_star-sum(a_f)
b_f = 1

sd_min_csi = 0.01 
sd_max_csi = 2
target_alpha_csi =   0.234
LAMB_csi = 0.7
EPS_csi = 0.01 
n0_zeta_csi = 1000 

SMOOT = "gardella"
WARP = "gamma-adaptation"

nburn = 2
niter = 5

nburn = 1000
niter = 500
##########################
# Data 
##########################
load("data_example.RData")

dati = dati_sim_3[[3]]

y = list(dati$y$y_to_use[,1:10], dati$y$y_to_use[,11:20])

n_groups = 2
n_per_group = c(10,10)
#########################
# Model input
#########################

params <- list(
  y = y,
  nburn = nburn,
  niter = niter,
  n_knots_beta = n_knots_beta,
  n_knots_gamma = n_knots_gamma,
  n_knots_phi = n_knots_phi,
  degree_beta = degree_beta,
  degree_gamma = degree_gamma,
  degree_phi = degree_phi,
  intercept_beta = intercept_beta,
  intercept_gamma = intercept_gamma,
  intercept_phi = intercept_phi,
  a_eps = a_eps,
  b_eps = b_eps,
  a_lambda = a_lambda,
  b_lambda = b_lambda,
  a_gamma = a_gamma,
  b_gamma = b_gamma,
  a_f = a_f,
  b_f = b_f,
  WARP = WARP,
  SMOOT = SMOOT,
  target_alpha = target_alpha_csi,
  sd_min = sd_min_csi,
  sd_max = sd_max_csi,
  LAMB = LAMB_csi,
  EPS = EPS_csi,
  n0_zeta = n0_zeta_csi
)

#########################
# Run the model 
#########################
Rcpp::sourceCpp("JARA_code.cpp")
set.seed(1)
result <- JARA_warping_GROUP(params)

#########################
# Plot the results
#########################
n1 <- 10
n2 <- 10
greens <- colorRampPalette(c("lightgreen", "darkgreen"))(n1)
reds <- colorRampPalette(c("pink", "darkred"))(n2)

ylim_plot_1 = c( min(dati$y$y_to_use[,1:10])  - 0.5, max(dati$y$y_to_use[,1:10]) +0.5 )
ylim_plot_2 = c( min(dati$y$y_to_use[,11:20]) - 0.5, max(dati$y$y_to_use[,11:20]) +0.5)

time = seq(0,1, len=300)

x11()
par(mfrow=c(1,2))
plot(time,y[[1]][,1], type='l', main='',
     col=greens[1], xlab='time', ylab='y',ylim=ylim_plot_1)
for(i in 2:n_per_group[1]){
  points(time,y[[1]][,i], type='l', col=greens[i])
}
plot(time,result$curves$y_star_smoot[[1]][[1]], type='l', main='',
     col=greens[1], xlab='', ylab='',ylim=ylim_plot_1)
for(i in 2:n_per_group[1]){
  points(time,result$curves$y_star_smoot[[1]][[i]], type='l', col=greens[i])
}

x11()
par(mfrow=c(1,2))
plot(time,y[[2]][,1], type='l', main='',
     col=reds[1], xlab='time', ylab='y',ylim=ylim_plot_2)
for(i in 2:n_per_group[1]){
  points(time,y[[2]][,i], type='l', col=reds[i])
}
plot(time,result$curves$y_star_smoot[[2]][[1]], type='l', main='',
     col=reds[1], xlab='', ylab='',ylim=ylim_plot_2)
for(i in 2:n_per_group[2]){
  points(time,result$curves$y_star_smoot[[2]][[i]], type='l', col=reds[i])
}
