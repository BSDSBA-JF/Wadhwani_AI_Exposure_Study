import os
import pandas as pd


def get_ability_exposure(
    worldbank_input_url: str,
    aioe_filename: str,
) -> dict[str, float]:
    """Calculate AI exposure scores for individual abilities."""
    ability_exposure_df = pd.read_excel(
        os.path.join(worldbank_input_url, aioe_filename),
        sheet_name="Appendix D",
        index_col=0,
    )

    ability_exposure_map = (
        ability_exposure_df
        .sum(axis=1)
        .to_dict()
    )

    # Standardize the ability name.
    if "Visual Color Determination" in ability_exposure_map:
        ability_exposure_map["Visual Color Discrimination"] = (
            ability_exposure_map["Visual Color Determination"]
        )

    return ability_exposure_map


def get_occupation_abilities(
    worldbank_input_url: str,
    code_title_map: dict[str, str],
) -> pd.DataFrame:
    """Load occupation abilities and impute missing occupation profiles."""
    abilities_df = pd.read_csv(
        os.path.join(worldbank_input_url, "abilities.csv")
    )

    abilities_df = (
        abilities_df
        .pivot(
            index=[
                "O*NET-SOC Code",
                "Title",
                "Element ID",
                "Element Name",
            ],
            columns="Scale Name",
            values="Data Value",
        )
        .reset_index()
    )

    # Identify occupation codes without ability data.
    all_possible_codes = set(code_title_map.keys())
    existing_codes = set(abilities_df["O*NET-SOC Code"])
    missing_codes = all_possible_codes - existing_codes

    # Create four-character SOC prefixes.
    first_n_prefix = 4
    abilities_df["SOC Prefix"] = (
        abilities_df["O*NET-SOC Code"]
        .str.replace("-", "", regex=False)
        .str[:first_n_prefix]
    )

    missing_codes_df = pd.DataFrame({
        "O*NET-SOC Code": sorted(missing_codes),
    })

    missing_codes_df["SOC Prefix"] = (
        missing_codes_df["O*NET-SOC Code"]
        .astype(str)
        .str.replace("-", "", regex=False)
        .str[:first_n_prefix]
    )

    # Calculate the mean ability profile within each SOC prefix.
    ability_means = (
        abilities_df
        .groupby(
            ["SOC Prefix", "Element ID", "Element Name"],
            as_index=False,
        )[["Importance", "Level"]]
        .mean()
    )

    # Impute missing occupation profiles using their SOC prefixes.
    abilities_to_impute = missing_codes_df.merge(
        ability_means,
        on="SOC Prefix",
        how="left",
    )

    abilities_to_impute["Title"] = (
        abilities_to_impute["O*NET-SOC Code"]
        .map(code_title_map)
    )

    # Combine observed and imputed occupation abilities.
    complete_abilities_df = pd.concat(
        [abilities_df, abilities_to_impute],
        ignore_index=True,
    )

    return complete_abilities_df


def calculate_AIOE(
    worldbank_input_url: str,
    aioe_filename: str,
    code_title_map: dict[str, str],
) -> pd.DataFrame:
    """Calculate raw and standardized AIOE scores for each occupation."""
    ability_exposure_map = get_ability_exposure(
        worldbank_input_url,
        aioe_filename,
    )

    complete_abilities_df = get_occupation_abilities(
        worldbank_input_url,
        code_title_map,
    )

    # Assign AI exposure scores to individual abilities.
    complete_abilities_df["Exposure"] = (
        complete_abilities_df["Element Name"]
        .map(ability_exposure_map)
    )

    # Calculate importance-weighted ability levels.
    complete_abilities_df["Importance-Weighted Level"] = (
        complete_abilities_df["Importance"]
        * complete_abilities_df["Level"]
    )

    complete_abilities_df["Exposure Weighted"] = (
        complete_abilities_df["Exposure"]
        * complete_abilities_df["Importance-Weighted Level"]
    )

    # Aggregate ability scores by occupation.
    aioe_df = (
        complete_abilities_df
        .groupby("O*NET-SOC Code")[
            ["Exposure Weighted", "Importance-Weighted Level"]
        ]
        .sum(min_count=1)
    )

    # Calculate raw AIOE.
    aioe_df["AIOE"] = (
        aioe_df["Exposure Weighted"]
        / aioe_df["Importance-Weighted Level"]
    )

    # Standardize AIOE across occupations.
    aioe_mean = aioe_df["AIOE"].mean()
    aioe_std = aioe_df["AIOE"].std()

    if pd.isna(aioe_std) or aioe_std == 0:
        aioe_df["Standardized AIOE"] = float("nan")
    else:
        aioe_df["Standardized AIOE"] = (
            aioe_df["AIOE"] - aioe_mean
        ) / aioe_std

    return aioe_df.reset_index()


def calculate_complementarity(
    worldbank_input_url: str,
    code_title_map: dict[str, str],
) -> pd.DataFrame:
    """Calculate occupation-level complementarity scores."""
    work_contexts_df = pd.read_csv(
        os.path.join(worldbank_input_url, "work_context.csv")
    )
    work_contexts_df = work_contexts_df[
        work_contexts_df["Scale ID"] == "CX"
    ].copy()
    work_contexts_df["Data Value"] = (
        work_contexts_df["Data Value"] - 1
    ) * 25

    # Create the degree of freedom.
    work_contexts_df["Element Name"] = (
        work_contexts_df["Element Name"].replace({
            "Degree of Automation": "Degree of Freedom"
        })
    )
    is_degree_freedom = (
        work_contexts_df["Element Name"] == "Degree of Freedom"
    )
    work_contexts_df.loc[is_degree_freedom, "Data Value"] = (
        100 - work_contexts_df.loc[is_degree_freedom, "Data Value"]
    )

    # Use only the relevant work contexts.
    relevant_contexts = [
        "Face-to-Face Discussions with Individuals and Within Teams",
        "Public Speaking",
        "Work Outcomes and Results of Other Workers",
        "Health and Safety of Other Workers",
        "Determine Tasks, Priorities and Goals",
        "Outdoors, Exposed to All Weather Conditions",
        "Physical Proximity",
        "Consequence of Error",
        "Frequency of Decision Making",
        "Degree of Freedom",
        "Freedom to Make Decisions",
    ]

    is_relevant_context = work_contexts_df["Element Name"].isin(
        relevant_contexts
    )
    work_contexts_df = work_contexts_df.loc[is_relevant_context]
    work_contexts_df = work_contexts_df[
        ["O*NET-SOC Code", "Title", "Element Name", "Data Value"]
    ].copy()

    # Get job zones.
    job_zones_df = pd.read_csv(
        os.path.join(worldbank_input_url, "job_zones.csv")
    )
    job_zones_df["Data Value"] = job_zones_df["Job Zone"] * 20
    job_zones_df["Element Name"] = "Skills"
    job_zones_df = job_zones_df[
        ["O*NET-SOC Code", "Title", "Element Name", "Data Value"]
    ].copy()

    # Combine work contexts and job zones.
    complementarity_df = pd.concat(
        [work_contexts_df, job_zones_df],
        ignore_index=True,
    )
    complementarity_df = complementarity_df.pivot(
        index=["O*NET-SOC Code", "Title"],
        columns="Element Name",
        values="Data Value",
    ).reset_index()

    # Keep occupations with complete complementarity data.
    complete_complementarity_df = complementarity_df.dropna().copy()

    # Create SOC prefixes.
    first_n_prefix = 4
    complete_complementarity_df["SOC Prefix"] = (
        complete_complementarity_df["O*NET-SOC Code"]
        .astype(str)
        .str.replace("-", "", regex=False)
        .str[:first_n_prefix]
    )

    # Identify completely missing SOC codes.
    all_possible_codes = set(code_title_map.keys())
    existing_codes = set(
        complete_complementarity_df["O*NET-SOC Code"]
    )
    missing_codes = all_possible_codes - existing_codes

    missing_codes_df = pd.DataFrame({
        "O*NET-SOC Code": list(missing_codes)
    })

    # Create SOC prefixes for missing codes.
    missing_codes_df["SOC Prefix"] = (
        missing_codes_df["O*NET-SOC Code"]
        .astype(str)
        .str.replace("-", "", regex=False)
        .str[:first_n_prefix]
    )

    # Define complementarity dimensions.
    complementarity_columns = relevant_contexts + ["Skills"]

    # Calculate mean complementarity profiles by SOC prefix.
    complementarity_means = (
        complete_complementarity_df
        .groupby("SOC Prefix")[complementarity_columns]
        .mean()
        .reset_index()
    )

    # Impute completely missing SOC codes.
    missing_complementarity = missing_codes_df.merge(
        complementarity_means,
        on="SOC Prefix",
        how="left",
    )

    complete_complementarity_df = pd.concat(
        [
            complete_complementarity_df,
            missing_complementarity,
        ],
        ignore_index=True,
    )

    # Remove temporary SOC Prefix column and restore titles.
    complete_complementarity_df.drop(
        columns="SOC Prefix",
        inplace=True,
    )
    complete_complementarity_df["Title"] = (
        complete_complementarity_df["O*NET-SOC Code"]
        .map(code_title_map)
    )

    # Calculate occupation-level complementarity.
    complete_complementarity_df["Complementarity"] = (
        complete_complementarity_df[complementarity_columns].mean(axis=1)
    )

    return complete_complementarity_df