from sackmann import get_data
import pymc as pm
from sklearn.preprocessing import LabelEncoder
import numpy as np


def create_arrays(
    start_year=1960,
    data_dir="./tennis_atp",
    include_qualifying_and_challengers=False,
    include_futures=False,
):

    df = get_data(
        data_dir,
        include_qualifying_and_challengers=include_qualifying_and_challengers,
        include_futures=include_futures,
    )

    rel_df = df[df["tourney_date"].dt.year >= start_year]

    encoder = LabelEncoder()

    encoder.fit(
        rel_df["winner_name"].values.tolist() + rel_df["loser_name"].values.tolist()
    )

    winner_ids = encoder.transform(rel_df["winner_name"])
    loser_ids = encoder.transform(rel_df["loser_name"])

    return {
        "winner_ids": winner_ids,
        "loser_ids": loser_ids,
        "player_encoder": encoder,
    }


def get_pymc_model(start_year=1960, data_dir="./tennis_atp"):

    arrays = create_arrays(start_year=start_year, data_dir=data_dir)

    n_players = len(arrays["player_encoder"].classes_)

    winner_ids = arrays["winner_ids"]
    loser_ids = arrays["loser_ids"]

    with pm.Model() as model:

        player_sd = pm.HalfNormal("player_sd", sigma=1.0)

        player_skills_raw = pm.Normal(
            "player_skills_raw", 0.0, sigma=1.0, shape=(n_players,)
        )

        player_skills = pm.Deterministic("player_skills", player_skills_raw * player_sd)
        logit_skills = player_skills[winner_ids] - player_skills[loser_ids]

        lik = pm.Bernoulli(
            "win_lik", logit_p=logit_skills, observed=np.ones(winner_ids.shape[0])
        )

    return model
