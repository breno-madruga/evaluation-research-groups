##########################################################################################################
################## Cleaning and Preprocessing the publications data and members' stats ###################
##########################################################################################################

########################################################################
# Importing required libraries.
import csv, pandas as pd, numpy as np, os
from scholarmetrics import hindex, gindex

# Creating the "prepared" directory.
os.makedirs("data_analysis_pipeline/data/prepared/", exist_ok=True)
########################################################################

########################################################################
# 1. Getting and checking the information dataset
########################################################################

########################################################################
# 1.1. Production data
########################################################################

# Importing the data.
df_data = pd.read_csv("data_analysis_pipeline/data/raw/manuscripts_group.csv", delimiter=",", header=0,
    dtype={"id": str, "pubmed_id": str})

# Describing the data.
print(df_data.describe())

# Checking some information about the data.
print(df_data.info())

########################################################################
# 1.2. Members' data and stats
########################################################################

# Importing the Members' data.
df_members = pd.read_csv("data_analysis_pipeline/data/raw/members_stats.csv", delimiter=",", header=0,
    dtype={"id": str})

# Checking some information about the data.
print(df_members.info())

########################################################################
# 2. Cleaning the dataframes
########################################################################

########################################################################
# 2.1. Members' data and stats
########################################################################

# Function to normalize the records of a feature.
def normalize_feature(row):
    if row:
        fields = list(row[0].keys())
        records = set([tuple([item[k] for k in fields]) for item in row])
        row = tuple([dict(zip(fields, item)) for item in records])
    return row

# Changing "NaN" values by "None" ones.
df_members.replace({np.nan: None}, inplace=True)

# Removing unnecessary columns of Members' data.
columns_deleted = ["initials", "surname", "indexed_name", "given_name", "eid", "list_eids_documents"]
df_members.drop(axis=1, columns=columns_deleted, inplace=True)

# Converting from the "str" type to the "list" type of some columns of Members data.
df_members.orcid = df_members.orcid.apply(lambda x: eval(x) if not pd.isnull(x) else None)
df_members.identifiers = df_members.identifiers.apply(lambda x: eval(x) if not pd.isnull(x) else None)
df_members.subject_areas = df_members.subject_areas.apply(lambda x: eval(x) if not pd.isnull(x) else None)
df_members.publication_range = df_members.publication_range.apply(lambda x: eval(x) if not pd.isnull(x) else None)
df_members.affiliation_current = df_members.affiliation_current.apply(lambda x: eval(x) if not pd.isnull(x) else None)
df_members.affiliation_history = df_members.affiliation_history.apply(lambda x: eval(x) if not pd.isnull(x) else None)

# Normalizing the features "subject_areas" and "affiliation_history".
df_members.subject_areas = df_members.subject_areas.apply(normalize_feature)
df_members.affiliation_history = df_members.affiliation_history.apply(normalize_feature)

# Defining the research line for each member.
member_per_research_line = {
    "Automation and Systems": [
        "ANDRÉ LAURINDO MAITELLI",
        "ANDRES ORTIZ SALAZAR",
        "CARLOS EDUARDO TRABUCO DOREA",
        "DIOMADSON RODRIGUES BELFORT",
        "FABIO MENEGHETTI UGULINO DE ARAUJO",
        "FLAVIO BEZERRA COSTA",
        "MANOEL FIRMINO DE MEDEIROS JUNIOR",
        "RICARDO LUCIO DE ARAUJO RIBEIRO",
        "SEBASTIAN YURI CAVALCANTI CATUNDA",
        "WALLACE MOREIRA BESSA"],

    "Computer Engineering": [
        "ADRIAO DUARTE DORIA NETO",
        "ALLAN DE MEDEIROS MARTINS",
        "DANIEL ALOISE",
        "IVANOVITCH MEDEIROS DANTAS DA SILVA",
        "LUIZ AFFONSO HENDERSON GUEDES DE OLIVEIRA",
        "LUIZ FELIPE DE QUEIROZ SILVEIRA",
        "LUIZ MARCOS GARCIA GONCALVES",
        "MARCELO AUGUSTO COSTA FERNANDES",
        "PABLO JAVIER ALSINA",
        "RICARDO ALEXSANDRO DE MEDEIROS VALENTIM",
        "SAMUEL XAVIER DE SOUZA"],

    "Telecommunication": [
        "ADAILDO GOMES D'ASSUNCAO",
        "ANTONIO LUIZ PEREIRA DE SIQUEIRA CAMPOS",
        "JOSE PATROCINIO DA SILVA",
        "VALDEMIR PRAXEDES DA SILVA NETO",
        "VICENTE ANGELO DE SOUSA JUNIOR"]
}

# Including the research line for each member and defining the permanent members.
for rl, members in member_per_research_line.items():
    df_members.loc[[member in members for member in df_members.complete_name], "research_line"] = rl
    df_members.loc[[member in members for member in df_members.complete_name], "is_permanent"] = True
df_members.loc[df_members.research_line.isnull(), "research_line"] = None
df_members["is_permanent"].fillna(False, inplace=True)

########################################################################
# 2.2. Production data
########################################################################

# Function to normalize the affiliations of the authors.
def normalize_affiliations(row):
    # Getting missing values within "author_affil" feature from "affiliations" one.
    if row.affiliations and row.author_affil:
        for pos, author in enumerate(row.author_affil):
            for affil in row.affiliations:
                if str(affil["id"]) and str(author["affil_id"]) and str(affil["id"]) in [af.strip()
                        for af in str(author["affil_id"]).split(",")]:
                    row.author_affil[pos]["affil_id"] = str(affil["id"])
                    row.author_affil[pos]["affiliation"] = affil["affiliation"]
                    if affil["country"] and not author["country"]:
                        row.author_affil[pos]["country"] = affil["country"]
                    elif affil["country"] != author["country"]:
                        row.author_affil[pos]["country"] = affil["country"]
    else:
        # Getting missing values within "affiliations" feature from "author_affil" one.
        if row.author_affil and not row.affiliations:
            affils = set([(str(author["affil_id"]), author["affiliation"], author["country"])
                        for author in row.author_affil
                        if author["affil_id"] or author["affiliation"] or author["country"]])
            if len(affils) > 0:
                keys = ["id", "affiliation", "country"]
                row.affiliations = tuple([dict(zip(keys, affil)) for affil in affils])
            else:
                row.affiliations = None
    return row

# Function to normalize the name of the authors.
def normalize_name_authors(row):
    if row.authors and row.author_affil:
        for pos, item in enumerate(row.authors):
            for author in list(row.author_affil):
                if str(item["id"]) == str(author["id"]):
                    row.authors[pos]["name"] = author["name"]
    return row

# Function to normalize the the authors and their affiliations.
def normalize_features(row):
    fields = {
        "authors": ["id", "name"],
        "affiliations": ["id", "affiliation", "country"],
        "affil": ["affil_id", "affiliation", "country"]
    }
    # Normalizing the authors.
    records = [tuple([item[f] for f in fields["authors"]]) for item in row.authors]
    if row.author_affil:
        records = set([*records, *[tuple([item[c] for c in fields["authors"]])
                                    for item in row.author_affil]])
    row.authors = tuple([dict(zip(fields["authors"], auth)) for auth in records])

    # Normalizing the affiliations.
    if row.affiliations:
        records = [tuple([item[c] for c in fields["affiliations"]])
                for item in row.affiliations]
        if row.author_affil:
            records = set([*records, *[tuple([item[c] for c in fields["affil"]])
                                    for item in row.author_affil]])
        row.affiliations = tuple([dict(zip(fields["affiliations"], affil))
                                for affil in records])
    return row

# Function to normalize the ID of some features.
def normalize_id_features(row):
    fields = ["id", "code", "affil_id"]
    features = row.index.tolist()
    for f in features:
        if pd.notnull(row[f]):
            row[f] = tuple([{k: str(item[k]) if k in fields else item[k] for k in item} for item in row[f]])
        else:
            row[f] = None
    return row

# Correcting the Scopus IDs of some articles.
ids = ["2-s2.0-85126083408", "2-s2.0-85112304772", "2-s2.0-85126284305", "2-s2.0-85110504864"]
df_data.id[df_data.id.isin(ids)] = df_data.id[df_data.id.isin(ids)].apply(lambda x: x.replace("2-s2.0-", ""))

# Checking the existence of invalid records.
print(df_data.id[df_data.id.notnull() & df_data.eid.isnull()])

# Removing the invalid record.
df_data = df_data[df_data.id.notnull() & df_data.eid.notnull()]

# Checking the result.
print(df_data.id[df_data.id.notnull() & df_data.eid.isnull()])

# Checking if there are duplicates by Scopus id.
print("Number of duplicated records:", df_data[df_data.id.duplicated()].id.size)

# Removing the duplicated records.
df_data.drop_duplicates("id", inplace=True)

# Checking if there are duplicates by Scopus id.
print("Number of duplicated records:", df_data[df_data.id.duplicated(keep=False)].id.size)

# Correcting the "vehicle_name" and "conference_name" columns.
df_data.loc[df_data.conference_name.notnull() & df_data.vehicle_name.isnull(), "vehicle_name"] = df_data.loc[
    df_data.conference_name.notnull() & df_data.vehicle_name.isnull(), "conference_name"]

# Normalizing some values of "vehicle_name" column.
df_data.loc[df_data.vehicle_name == "Controle y Automacao", "vehicle_name"] = "Controle and Automacao"

# Removing unnecessary columns of Production data.
columns_deleted = ["doi", "pii", "pubmed_id", "description", "conf_location", "conference_name",
                   "vehicle_address", "title_edition", "publisher"]
df_data.drop(axis=1, columns=columns_deleted, inplace=True)

# Changing the type of "publication_date" column.
df_data.publication_date = pd.to_datetime(df_data.publication_date, format="%Y-%m-%d")

# Changing the type of "citation_num" and "ref_count" columns.
df_data.citation_num = df_data.citation_num.apply(lambda x: int(x) if not pd.isnull(x) else None)
df_data.ref_count = df_data.ref_count.apply(lambda x: int(x) if not pd.isnull(x) else None)

# Converting from the "str" type to the "list" type of some columns of Production data.
df_data.replace({np.nan: None}, inplace=True)
df_data.auth_keywords = df_data.auth_keywords.apply(lambda x: eval(x) if not pd.isnull(x) else None)
df_data.index_terms = df_data.index_terms.apply(lambda x: eval(x) if not pd.isnull(x) else None)
df_data.affiliations = df_data.affiliations.apply(lambda x: eval(x) if not pd.isnull(x) else None)
df_data.subject_areas = df_data.subject_areas.apply(lambda x: eval(x) if not pd.isnull(x) else None)
df_data.authors = df_data.authors.apply(lambda x: eval(x) if not pd.isnull(x) else None)
df_data.author_affil = df_data.author_affil.apply(lambda x: eval(x) if not pd.isnull(x) else None)
df_data.references = df_data.references.apply(lambda x: eval(x) if not pd.isnull(x) else None)

# Updating the "members_name" column for each work.
df_data.members_name = [tuple(df_members.complete_name[
        [len(set(ids).intersection(set([str(author["id"]) for author in authors]))) > 0
         for ids in df_members.identifiers.values]
].values) for authors in df_data.authors.values]

# Creating the "number_members" column for each work.
df_data["number_members"] = [df_members.complete_name[
        df_members.complete_name.isin(member) & df_members.is_permanent].size
                for member in df_data.members_name.values]

# Setting the country for the brazilian affiliations without this feature.
ids = ["0031104541", "26944494548", "51749115113", "67649946788",
       "39749115635", "77952509269"]
df_data.affiliations[df_data.id.isin(ids)] = [tuple([{**affil, "country": "Brazil"} \
       if not affil["country"] else affil for affil in affils])
    for affils in df_data.affiliations[df_data.id.isin(ids)]]

# Removing the duplicated authors within the same record.
df_data.author_affil[df_data.id == "33845809181"] = df_data.author_affil[
    df_data.id == "33845809181"].apply(lambda x: x[:4])

# Defining the correct affiliation for record whose id is "79955775766".
df_data.author_affil[df_data.id == "79955775766"] = df_data.author_affil[
    df_data.id == "79955775766"].apply(lambda x: tuple([*x[:2], x[-1]]))

# Defining the correct affiliation for record whose id is "85020791355".
df_data.affiliations[df_data.id == "85020791355"] = df_data.affiliations[
    df_data.id == "85020791355"].apply(lambda x: tuple([*df_data.affiliations[
        df_data.id == "84947312273"].iloc[0], {"id": "60010758",
        "affiliation": "Université de Mons", "country": "Belgium"}]))
df_data.author_affil[df_data.id == "85020791355"] = df_data.loc[df_data.id == "85020791355",
    ["affiliations", "author_affil"]].apply(lambda row: tuple([
        {**author, "affil_id": str(row.affiliations[0]["id"]) \
            if author["affiliation"] == "UFRN" else str(row.affiliations[1]["id"])}
        for author in row.author_affil]), axis=1)

# Defining the correct affiliation for record whose id is "85026837976".
df_data.author_affil[df_data.id == "85026837976"] = df_data.author_affil[
    df_data.id == "85026837976"].apply(lambda x: tuple([*x[:3], *[{**affil,
        "affil_id": "60011324"} for affil in x[3:]]]))

# Defining the correct affiliation for record whose id is "0036949197".
temp = [{"id": "60003709", "affiliation": "Universidade Federal de Campina Grande",
    "country": "Brazil"}, {"id": "60011324", "affiliation": "Universidade Federal da Paraiba",
    "country": "Brazil"}]
df_data.affiliations[df_data.id == "0036949197"] = df_data.affiliations[
    df_data.id == "0036949197"].apply(lambda x: tuple([*x, *temp]))
temp = {"56249460300": "60011324", "7202634615": "60003709", "7201494953": "60011324"}
df_data.author_affil[df_data.id == "0036949197"] = df_data.author_affil.loc[
    df_data.id == "0036949197"].apply(lambda x: tuple([*x[:4], *[{**affil,
        "affil_id": temp[str(affil["id"])]} for affil in x[4:]]]))

# Defining the correct affiliation for record whose id are "85043396160", "0032302636", "85114962520".
temp = {"id": "60023857", "affiliation": "Universidade Federal do Rio Grande do Norte", "country": "Brazil"}
df_data.affiliations[df_data.id.isin(["85043396160", "0032302636", "85114962520"])] = df_data.affiliations[
    df_data.id.isin(["85043396160", "0032302636", "85114962520"])].apply(
        lambda x: tuple([affil if str(affil["id"]) not in {"112589976", "126896742"} \
            else temp for affil in x]))
temp = {"112589976": "60023857", "126896742": "60023857"}
df_data.author_affil[df_data.id.isin(["85043396160", "0032302636", "85114962520"])] = df_data.author_affil.loc[
    df_data.id.isin(["85043396160", "0032302636", "85114962520"])].apply(lambda x: tuple([{**affil,
        "affil_id": temp[str(affil["affil_id"])] if str(affil["affil_id"]) in temp else str(affil["affil_id"])}
            for affil in x]))

# Correcting the list of authors and their affiliations for record whose id is "84942546461".
temp = {"id": "57040578700", "name": "Ádller De O. Guimarães"}
df_data.authors[df_data.id == "84942546461"] = df_data.authors[
    df_data.id == "84942546461"].apply(lambda x: tuple([*x[:2], temp]))
temp = {**temp, "affil_id": None, "affiliation": None, "country": None}
df_data.author_affil[df_data.id == "84942546461"] = df_data.author_affil[
    df_data.id == "84942546461"].apply(lambda x: tuple([*x[:3], *[{**temp,
        "affil_id": affil} for affil in ["60023857", "114536011"]]]))

# Defining the correct affiliation for the some records.
df_data.author_affil[df_data.id == "84867959509"] = df_data.author_affil[
    df_data.id == "84867959509"].apply(lambda x: tuple([*x[:-1], {**x[-1],
        "affil_id": str(x[-2]["affil_id"]), "affiliation": x[-2]["affiliation"],
        "country": x[-2]["country"]}]))
df_data.author_affil[df_data.id == "22744442063"] = df_data.author_affil[
    df_data.id == "22744442063"].apply(lambda x: tuple([*x[:2], {**x[2],
        "affil_id": "60003709"}, *x[3:6], {**x[6], "affil_id": "60003709"}]))
df_data.author_affil[df_data.id == "85081616543"] = df_data.author_affil[
    df_data.id == "85081616543"].apply(lambda x: tuple([*x[:-2],
        *df_data.author_affil[df_data.id == "85081591459"].item()[-2:]]))

# Defining the correct affiliation for the some records.
ids = ["84920913312", "85050497831"]
df_data.affiliations[df_data.id.isin(ids)] = df_data.affiliations[
    df_data.id.isin(ids)].apply(lambda x: df_data.affiliations[
        df_data.id == "84947312273"].item())
df_data.author_affil[df_data.id.isin(ids)] = df_data.loc[df_data.id.isin(ids),
    ["affiliations", "author_affil"]].apply(lambda row: tuple([
        {**author, "affil_id": str(row.affiliations[0]["id"])}
        for author in row.author_affil]), axis=1)

# Removing the editor from the list of authors.
temp = ["56028680000", "35732489900"]
ids = ["85062323680", "84927799972"]
df_data.author_affil[df_data.id.isin(ids)] = df_data.author_affil[
    df_data.id.isin(ids)].apply(lambda x: tuple(
        [auth for auth in x if str(auth["id"]) not in temp]))
df_data.authors[df_data.id.isin(ids)] = df_data.authors[
    df_data.id.isin(ids)].apply(lambda x: tuple(
        [auth for auth in x if str(auth["id"]) not in temp]))

# Defining the alternative identifier to the affiliations without their IDs or the null ones.
idx = list(set([idx for idx, row in df_data.author_affil[df_data.author_affil.notnull()].iteritems()
                for item in row if not eval(str(item["affil_id"]))]))
df_data.author_affil[idx] = [
    tuple([{**affil, "affil_id": str(hash(affil["affiliation"])) \
                if not eval(str(affil["affil_id"])) and affil["affiliation"] else \
                eval(str(affil["affil_id"])) if not eval(str(affil["affil_id"])) and not affil["affiliation"] \
                else affil["affil_id"]} for affil in row])
    for row in df_data.author_affil[idx]]

# Applying the "normalize_affiliations" function to the data.
df_data[["affiliations", "author_affil"]] = df_data[
    ["affiliations", "author_affil"]].apply(normalize_affiliations, axis=1)

# Removing duplicates within the list of affiliations and authors.
df_data.author_affil = [
    set([(str(au["id"]), au["name"], str(au["affil_id"]),
        au["affiliation"], au["country"]) for au in row]) if row else None
    for row in df_data.author_affil]
df_data.author_affil = [tuple([dict(zip(
        ["id", "name", "affil_id", "affiliation", "country"], au)) for au in row]) if row else None
    for row in df_data.author_affil]

# Applying the "normalize_name_authors" function to the data.
df_data[["authors", "author_affil"]] = df_data[["authors", "author_affil"]].apply(
    normalize_name_authors, axis=1)

# Applying the "normalize_features" function to the data.
df_data[["authors", "affiliations", "author_affil"]] = df_data[
    ["authors", "affiliations", "author_affil"]].apply(
        normalize_features, axis=1)

# Defining the "year" column from the "publication_date" column.
df_data["year"] = pd.DatetimeIndex(df_data.publication_date).year

# Defining the "month" column from the "publication_date" column.
df_data["month"] = pd.DatetimeIndex(df_data.publication_date).month

# Updating the members' h-index from their production.
df_members["h_index"] = df_members.complete_name.apply(lambda x: hindex(
    df_data.citation_num[[x in members for members in df_data.members_name]].values))

# Creating the members' g-index from their production.
df_members["g_index"] = df_members.complete_name.apply(lambda x: gindex(
    df_data.citation_num[[x in members for members in df_data.members_name]].values))

# Defining the h2-index of research group.
df_members["h2_index"] = hindex(df_members.h_index[df_members.is_permanent].values)

# Applying the "normalize_id_features" function to the data.
df_data[["authors", "affiliations", "author_affil", "subject_areas", "references"]] = df_data[
    ["authors", "affiliations", "author_affil", "subject_areas", "references"]].apply(
        normalize_id_features, axis=1)

# Defining the analysis' period.
period = list(range(2010, 2023))

# Filtering the data.
df_data = df_data[df_data.year.isin(period)]

########################################################################
# 3. Fixing the inconsistences of dataframe
########################################################################

# Creating the dictionary with the old and new ISSNs.
issn = {"07168756": "07180764", "14148862": "1984557X", "01959271": "18666892",
        "0103944X": "19834071", "16875877": "16875869", "1558187X": "00189375",
        "16784804": "01046500", "09746870": "09713514", "10459227": "2162237X",
        "16875249": "16875257", "14148862": "1984557X", "23203765": "22788875",
        "23090413": "03757765", "21791073": "21791074", "17518644": "17518652",
        "15498328": "15580806", "19374208": "08858977", "18070302": "01018205",
        "19842538": "1984252X", "15730484": "09208542", "15728080": "09295585",
        "19255810": "14801752", "14698668": "02635747", "01034308": "21752745",
        "07437315": "10960848", "15730409": "09210296", "10947167": "15411672",
        "23174609": "23190566", "16155297": "16155289", "2195268X": "21952698",
        "16779649": "22366733", "10834419": "21682267", "19430671": "19430663",
        "1558187X": "00189375", "11092777": "22242678", "16771966": "21798451",
        "16875257": "16875249", "15167399": "21791074", "15173151": "24464740",
        "13502379": "17518652"}

# Updating the old ISSN to the new ISSN.
for issn_old, issn_new in issn.items():
    df_data.issn.loc[df_data.issn.notnull() & df_data.issn.str.contains(issn_old, na=False) &
                     ~df_data.issn.str.contains(issn_new, na=False)] = df_data.issn.loc[
                     df_data.issn.notnull() & df_data.issn.str.contains(issn_old, na=False) &
                     ~df_data.issn.str.contains(issn_new, na=False)].apply(
                         lambda x: "{} {}".format(x, issn_new))

########################################################################
# 4. Exporting the data
########################################################################

# Saving the production data.
columns = ["members_name", "id", "title", "abstract", "citation_num", "auth_keywords", "index_terms",
           "vehicle_name", "affiliations", "subject_areas", "authors", "author_affil",
           "year", "month", "ref_count", "references"]
df_data[columns].to_csv("data_analysis_pipeline/data/prepared/production_members_final.csv", index=False, quoting=csv.QUOTE_ALL)

# Saving the members' data.
columns = ["complete_name", "identifiers", "h_index", "is_permanent", "research_line", "subject_areas",
           "citation_count", "document_count", "coauthor_count", "affiliation_current", "affiliation_history"]
df_members[columns].to_csv("data_analysis_pipeline/data/prepared/members_stats_final.csv", index=False, quoting=csv.QUOTE_ALL)