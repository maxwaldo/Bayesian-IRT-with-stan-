data {
  int<lower=1> N;
  int<lower=1> I;
  int<lower=1> N_d;
  int<lower=1> N_nd;
  int<lower=1> J;
  int<lower=0> y[N];
  real<lower=0, upper=1> y_c[N_d];
  int<lower=0> i[N];
  int<lower=0> i_d[N_d];
  int<lower=0> i_nd[N_nd];
  int<lower=0> j[N];
}

transformed data {
  real yy[N_d];
  yy = logit(y_c);
}

parameters {
  vector[I] alpha;
  vector[I] beta;
  vector[J+1] theta;
  vector<lower=0>[I] sigma; 
}

transformed parameters {
  real<lower=0, upper=1> p_binary[N];
  real<lower=negative_infinity(), upper=positive_infinity()> p_continuous[N_d];

  for (n in 1:N) {
    p_binary[n] = inv_logit(theta[j[n]] * beta[i[n]] - alpha[i[n]]);
  }
  for (n in 1:N_d) {
    p_continuous[n] = (theta[J+1] * beta[i_d[n]] - alpha[i_d[n]]) * sigma[i_d[n]];
  }
}

model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 5);
  theta ~ normal(0, 1);
  sigma ~ cauchy(0, 2.5);
  
  y ~ bernoulli(p_binary);
  yy ~ normal(p_continuous, 1);
}

generated quantities {
  real pred[I];
  for(n in 1:I) {
    pred[n] = inv_logit(theta[J+1] * beta[n] - alpha[n]);
  }
}

