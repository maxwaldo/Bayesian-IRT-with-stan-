data {
  int<lower=1> N;
  int<lower=1> I;
  int<lower=1> J;
  int<lower=1> T;
  int<lower=0> y[N];
  int<lower=1> i[N];
  int<lower=1> j[N];
  int<lower=1> t[N];
  int<lower=1> N_c;
  int<lower=1> J_c;
  real<lower=0> y_c[N_c];
  int<lower=1> i_c[N_c];
  int<lower=1> j_c[N_c];
  int<lower=1> t_c[N_c];
}

transformed data{
  real yy[N_c];
  yy = logit(y_c);
}

parameters {
  vector[I] alpha; // Difficulty parameter
  vector[I] beta;  // Discrimination parameter
  vector[J+J_c] theta;  // ideal position of actor j at time t;
  vector[J+J_c] time[T-1]; // random time variable.
  vector<lower=0>[I] sigma;
  vector<lower=0, upper =1>[J+J_c] time_var;
}

transformed parameters {
  real<lower=0, upper=1> p_binary[N];
  vector[J+J_c] theta2[T];
  real<lower=negative_infinity(), upper=positive_infinity()> p_continuous[N_c];
 
  for(n in 1:T) {
    if(n==1) {
      theta2[n] = theta;
    } if(n>1) {
      theta2[n] = theta2[n-1] + time[n-1] .* time_var;
    }
  }
  
  for (n in 1:N) {
    p_binary[n] = inv_logit(theta2[t[n], j[n]] * beta[i[n]] + alpha[i[n]]);
  }
  for (n in 1:N_c) {
    p_continuous[n] = (theta2[t_c[n], J+j_c[n]] * beta[i_c[n]] + alpha[i_c[n]]) * sigma[i_c[n]];
  }
}
  
model {
  
  target += normal_lpdf(beta| 0, 5);
  target += normal_lpdf(alpha| 0, 2);
  target += cauchy_lpdf(sigma| 0, 2.5);
  target += exponential_lpdf(time_var| 10);
  target += normal_lpdf(theta| 0, 1);
  
  // at time 1, no difference between theta and theta2
  
  for (n in 1:T-1) {
    time[n] ~ normal(0, 1);
  }
  y ~ bernoulli(p_binary);
  target += normal_lpdf(yy| p_continuous, 1);
}

