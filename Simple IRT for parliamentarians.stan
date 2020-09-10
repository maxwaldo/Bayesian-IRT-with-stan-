data {
  int<lower=1> N; // Number of observation
  int<lower=1> I; // Number of Items
  int<lower=1> N_d; // Number of Direct democratic votes
  int<lower=1> N_nd; // Number of non-Direct democratic items
  int<lower=1> J; // Number of actors (parliamentarians)
  int<lower=0> y[N]; // Binary outcome
  real<lower=0, upper=1> y_c[N_d]; // Continuous outcome (population)
  int<lower=0> i[N]; // item list for each observation 
  int<lower=0> i_d[N_d]; // items list for continuous observation
  int<lower=0> i_nd[N_nd]; // item list for non-direct democratic votes (what we would like ot predict)
  int<lower=0> j[N]; // List of actor for observation 1:N
}


transformed data {
  real yy[N_d];
  yy = logit(y_c); // Logit transformation with the continuous variables (so we have it from -infinity to +infinity)
}

parameters {
  vector[I] alpha; // Difficulty of items
  vector[I] beta; // Discrimination of items
  vector[J+1] theta; // Ability of actor
  vector<lower=0>[I] sigma; 
}

transformed parameters {
  real<lower=0, upper=1> p_binary[N];
  real<lower=negative_infinity(), upper=positive_infinity()> p_continuous[N_d];

  // This is for the non-centric parametrization
  for (n in 1:N) {
    p_binary[n] = inv_logit(theta[j[n]] * beta[i[n]] - alpha[i[n]]);
  }
  for (n in 1:N_d) {
    p_continuous[n] = (theta[J+1] * beta[i_d[n]] - alpha[i_d[n]]) * sigma[i_d[n]]; // J+1 represents the swiss population
  }
}

model {
  alpha ~ normal(0, 10); // Priors for the difficulty of items
  beta ~ normal(0, 10); // Priors for the 
  theta ~ normal(0, 1); // Prior for the actor position on the latent axis
  sigma ~ cauchy(0, 2.5);
  
  y ~ bernoulli(p_binary); // Model for the binary IRT
  yy ~ normal(p_continuous, 1);  // Model for the Continuous IRT
}

generated quantities {
  real pred[I];
  for(n in 1:I) {
    pred[n] = inv_logit(theta[J+1] * beta[n] - alpha[n]); // Predicts populare support for partliamentary projects. 
  }
}

