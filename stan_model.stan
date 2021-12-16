data {
    int n_matches;
    int n_players;
    
    int winner_ids[n_matches];
    int loser_ids[n_matches];
}
transformed data {
    int outcomes[n_matches];
    outcomes = rep_array(1, n_matches);
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
    player_skills_raw ~ std_normal();
    player_sd ~ normal(0, 1);
    
    outcomes ~ bernoulli_logit(player_skills[winner_ids] - player_skills[loser_ids]);
    
}
