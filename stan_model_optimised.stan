data {
    int n_matches;
    int n_players;
    
    int winner_ids[n_matches];
    int loser_ids[n_matches];
}
parameters {
    vector[n_players] player_skills_raw;
    real<lower=0> player_sd;
}
transformed parameters {
    vector[n_players] player_skills;
    player_skills = player_skills_raw * player_sd;
}
model {   
    vector[n_matches] mu;

    player_skills_raw ~ std_normal();
    player_sd ~ normal(0, 1);

    // As suggested by Bob Carpenter, do not vectorise.
    for (n in 1:n_matches) {
	mu[n] = player_skills[winner_ids[n]] - player_skills[loser_ids[n]];
    }

    1 ~ bernoulli_logit(mu);
    
}
