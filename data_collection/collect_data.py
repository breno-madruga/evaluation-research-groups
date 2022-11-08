##########################################################################################################
################ Collecting the chosen research group production data and its members' stats #############
##########################################################################################################

# For collecting the publications and researchers' stats, we used the "pybliometrics" library.
# It is avaliable on this [link](https://pypi.org/project/pybliometrics/).

########################################################################
# Uncomment to install the library.
# %pip install pybliometrics
########################################################################

########################################################################
# Importing the required libraries.
import csv, pandas as pd, numpy as np
from datetime import datetime
from pybliometrics.scopus import config, AbstractRetrieval, AuthorRetrieval
from pybliometrics.scopus.utils import create_config
from pybliometrics.scopus.exception import Scopus404Error, Scopus429Error, Scopus500Error
from urllib.error import HTTPError
########################################################################

########################################################################
# 1. Getting the data from "pybliometrics" library
########################################################################

# Uncomment to config on the first time.
# create_config()

# Setting the Scopus API Key.
_keys = [">>> PUT HERE YOUR SCOPUS API KEYS <<<"]
config["Authentication"]["APIKey"] = _keys.pop()

########################################################################
# 1.1. Getting the researchers' data and stats
########################################################################

# Scopus' IDs of Team's members.
scopus_ids = {
    ############################################################################
    ######################### Senior Members (2021-2022) #######################
    ############################################################################
    # Automation and Systems.
    "ANDRÉ LAURINDO MAITELLI": ["57188958500", "6602549353"],
    "ANDRES ORTIZ SALAZAR": ["7102422753"],
    "CARLOS EDUARDO TRABUCO DOREA": ["7004096252"],
    "DIOMADSON RODRIGUES BELFORT": ["24775112000", "57214364356"],
    "FABIO MENEGHETTI UGULINO DE ARAUJO": ["35316777100", "36655735300", "7004017586", "57223356933"],
    "FLAVIO BEZERRA COSTA": ["22333585200", "57217339817"],
    "MANOEL FIRMINO DE MEDEIROS JUNIOR": ["56525260900", "55360722000", "57211288717", "6603603627",
                                          "57209350817"],
    "RICARDO LUCIO DE ARAUJO RIBEIRO": ["57193949292", "7202534599", "57210817847"],
    "SEBASTIAN YURI CAVALCANTI CATUNDA": ["6602674444", "57195273898", "24476172600"],
    "WALLACE MOREIRA BESSA": ["6506394388"],

    # Computer Engineering.
    "ADRIAO DUARTE DORIA NETO": ["7102618949", "57211278113", "36833514600", "57210457821",
                                 "7102601960", "57197245198", "57215344419", "57209283982",
                                 "57212565466"],
    "ALLAN DE MEDEIROS MARTINS": ["55415511600", "57214364589"],
    "DANIEL ALOISE": ["12759644500"],
    "IVANOVITCH MEDEIROS DANTAS DA SILVA": ["36537969500"],
    "LUIZ AFFONSO HENDERSON GUEDES DE OLIVEIRA": ["14631495300", "24801938700", "22136014400",
                                                  "35761038800", "57191609276"],
    "LUIZ FELIPE DE QUEIROZ SILVEIRA": ["56363591700", "57214722886"],
    "LUIZ MARCOS GARCIA GONCALVES": ["8843298100"],
    "MARCELO AUGUSTO COSTA FERNANDES": ["7202947679", "57220883161"],
    "PABLO JAVIER ALSINA": ["6603627735"],
    "RICARDO ALEXSANDRO DE MEDEIROS VALENTIM": ["35606294400", "55555939300", "57192702752", "57204037954",
                                                "57200417054"],
    "SAMUEL XAVIER DE SOUZA": ["57212483740", "7801645972", "57221624129", "57221624125"],

    # Telecommunication.
    "ADAILDO GOMES D'ASSUNCAO": ["7004474343", "57208902469", "57213002002"],
    "ANTONIO LUIZ PEREIRA DE SIQUEIRA CAMPOS": ["16309023400", "57211219391"],
    "JOSE PATROCINIO DA SILVA": ["8776554000", "57212234960", "57212234942"],
    "VALDEMIR PRAXEDES DA SILVA NETO": ["55908415500", "56808964300", "57197869420"],
    "VICENTE ANGELO DE SOUSA JUNIOR": ["6603844872", "57209857067", "57216592462"],

    ############################################################################
    ########################### Post PhD (2021-2022) ###########################
    ############################################################################
    "TIAGO TAVARES LEITE BARROS": ["56369616300", "57213725905", "57206663392"],
    "MELINDA CESIANARA SILVA DA CRUZ": ["57213933794"],
    "IGOR GADÊLHA PEREIRA": ["57195577525", "56545245300"],
    "ITALO AUGUSTO SOUZA DE ASSIS": ["56942487700", "57213824515"],
    "JORIS GUERIN": ["57193013593", "57224522656", "57214782306"],
    "LUCAS COSTA PEREIRA CAVALCANTE": ["56428717400", "57215564751"],
    "ROBSON DE MACEDO FILHO": ["57214269128", "57234383400"]
}

# Function to collect members' data and stats.
def collect_data_members(ids_scopus):
    data = []
    for author, list_key in ids_scopus.items():
        record = {"complete_name": author, "identifiers": list_key}
        visited = set()
        error = True
        while error:
            try:
                list_key = list(set(list_key).difference(visited))
                for key in list_key:
                    print(author, "-", key)
                    au = AuthorRetrieval(key, refresh=True)
                    # Checking if this author has a paper.
                    if pd.notnull(au.document_count) and str(au.identifier) not in visited:
                        # Attributes non-updated.
                        if "h_index" not in record or int(au.h_index if au.h_index else "0") >= record["h_index"]:
                            record["h_index"] = int(au.h_index if au.h_index else "0")
                            record["initials"] = au.initials
                            record["surname"] = au.surname
                            record["indexed_name"] = au.indexed_name
                            record["id"] = str(au.identifier)
                            record["given_name"] = au.given_name
                            record["eid"] = str(au.eid)
                            record["affiliation_current"] = tuple([{
                                    "id": str(affil.id) if affil and str(affil.id) else None,
                                    "affiliation": affil.preferred_name if affil and affil.preferred_name else None,
                                    "country": affil.country if affil and affil.country else None}
                                for affil in au.affiliation_current]) if au.affiliation_current else None

                        # Attributes as lists.
                        if "affiliation_history" not in record and au.affiliation_history:
                            record["affiliation_history"] = tuple([{
                                    "id": str(affil.id) if affil and str(affil.id) and affil.type == "parent" else \
                                        affil.parent if affil and affil.parent and affil.type == "dept" else None,
                                    "affiliation": affil.preferred_name if affil and affil.preferred_name and \
                                        affil.type == "parent" else affil.parent_preferred_name if affil and \
                                        affil.parent_preferred_name and affil.type == "dept" else None,
                                    "country": affil.country if affil and affil.country else None}
                                for affil in au.affiliation_history])
                        elif au.affiliation_history:
                            record["affiliation_history"] = tuple([*record["affiliation_history"], *[{
                                    "id": str(affil.id) if affil and str(affil.id) and affil.type == "parent" else \
                                        affil.parent if affil and affil.parent and affil.type == "dept" else None,
                                    "affiliation": affil.preferred_name if affil and affil.preferred_name and \
                                        affil.type == "parent" else affil.parent_preferred_name if affil and \
                                        affil.parent_preferred_name and affil.type == "dept" else None,
                                    "country": affil.country if affil and affil.country else None}
                                for affil in au.affiliation_history]])

                        if "list_eids_documents" not in record and au.get_document_eids(view="COMPLETE", refresh=True):
                            record["list_eids_documents"] = list(set(au.get_document_eids(view="COMPLETE", refresh=True)))
                        elif au.get_document_eids(view="COMPLETE", refresh=True):
                            record["list_eids_documents"] = list(set(au.get_document_eids(view="COMPLETE", refresh=True)).union(
                                set(record["list_eids_documents"])))

                        if "subject_areas" not in record and au.subject_areas:
                            record["subject_areas"] = tuple([{"area": sub_area.area, "code": str(sub_area.code),
                                                              "abbrev": sub_area.abbreviation}
                                                             for sub_area in au.subject_areas])
                        elif au.subject_areas:
                            record["subject_areas"] = tuple([*[{"area": sub_area.area, "code": str(sub_area.code),
                                                                "abbrev": sub_area.abbreviation}
                                                               for sub_area in au.subject_areas], *record["subject_areas"]])

                        if "orcid" not in record and au.orcid:
                            record["orcid"] = [au.orcid]
                        elif au.orcid:
                            record["orcid"] = tuple(set([*record["orcid"], au.orcid]))

                        if "publication_range" not in record and au.publication_range:
                            record["publication_range"] = tuple(list(au.publication_range))
                        elif au.publication_range:
                            record["publication_range"] = tuple(
                                [min(int(record["publication_range"][0]), int(au.publication_range[0])),
                                max(int(record["publication_range"][1]), int(au.publication_range[1]))])

                        if "citation_count" not in record and int(au.citation_count):
                            record["citation_count"] = int(au.citation_count)
                        elif int(au.citation_count):
                            record["citation_count"] += int(au.citation_count)

                        if "cited_by_count" not in record and int(au.cited_by_count):
                            record["cited_by_count"] = int(au.cited_by_count)
                        elif int(au.cited_by_count):
                            record["cited_by_count"] += int(au.cited_by_count)

                        if "document_count" not in record and int(au.document_count):
                            record["document_count"] = int(au.document_count)
                        elif int(au.document_count):
                            record["document_count"] += int(au.document_count)

                        if "coauthor_count" not in record and au.coauthor_count:
                            record["coauthor_count"] = int(au.coauthor_count)
                        elif au.coauthor_count:
                            record["coauthor_count"] += int(au.coauthor_count)

                    # Adding the author's ID already visited.
                    visited.add(str(key))
                    if str(key) != record["id"]:
                        visited.add(record["id"])
                # Stopping the loop when all author to be visited.
                if len(list_key) == 0:
                    error = False
            except (Scopus404Error, Scopus500Error, HTTPError, KeyError) as e:
                print("Error:", record["complete_name"], "-", key)
                visited.add(str(key))
                if "id" in record and record["id"] not in visited and str(key) != record["id"]:
                    visited.add(record["id"])
            except Scopus429Error as e:
                # Removing the last item in _keys to assign it as new API key.
                config["Authentication"]["APIKey"] = _keys.pop()
                if len(_keys) == 0:
                    raise e
        data.append(record)
    return data

# Getting data.
data = collect_data_members(scopus_ids)

# Saving the data into CSV file.
pd.DataFrame(data).to_csv("../data_analysis_pipeline/data/raw/members_stats.csv", index=False, quoting=csv.QUOTE_ALL)

########################################################################
# 1.2. Getting the publications' data from list of EIDs
########################################################################

# Function to collect manuscripts' data.
def collect_data_manuscripts(data_members):
    data = []
    for item in data_members:
        print(item["complete_name"])
        for key in item["list_eids_documents"]:
            record = {"member_name": item["complete_name"]}
            error = True
            while error:
                try:
                    paper = AbstractRetrieval(key, id_type="eid", view="FULL", refresh=True)
                    error = False
                    # Basic Attributes.
                    record["id"] = str(paper.identifier)
                    record["doi"] = paper.doi
                    record["eid"] = str(paper.eid)
                    record["pii"] = paper.pii
                    record["pubmed_id"] = paper.pubmed_id
                    record["title"] = paper.title
                    record["abstract"] = paper.abstract
                    record["description"] = paper.description
                    record["publication_date"] = datetime.strptime(paper.coverDate, "%Y-%m-%d").date() \
                                                 if paper.coverDate else None
                    record["citation_num"] = paper.citedby_count
                    record["language"] = paper.language
                    record["production_type"] = paper.aggregationType
                    record["source_type"] = paper.srctype
                    record["auth_keywords"] = tuple(paper.authkeywords) if paper.authkeywords else None
                    record["index_terms"] = tuple(paper.idxterms) if paper.idxterms else None
                    record["issn"] = paper.issn

                    try:
                        record["isbn"] = " ".join(paper.isbn) if type(paper.isbn) == tuple else paper.isbn
                    except TypeError:
                        record["isbn"] = None

                    # Conference and/or Journals data.
                    record["conf_location"] = paper.conflocation

                    try:
                        record["conference_name"] = paper.confname
                    except AttributeError:
                        record["conference_name"] = None

                    record["vehicle_name"] = paper.publicationName
                    record["vehicle_address"] = paper.publisheraddress
                    record["title_edition"] = paper.issuetitle
                    record["publisher"] = paper.publisher

                    # Affiliation.
                    record["affiliations"] = tuple(
                        [{"id": str(affil.id) if affil and str(affil.id) else None,
                        "affiliation": affil.name if affil and affil.name else None,
                        "country": affil.country if affil and affil.country else None}
                        for affil in paper.affiliation]) if paper.affiliation else None

                    # Subject Areas.
                    record["subject_areas"] = tuple([{"area": area.area, "code": str(area.code),
                                                      "abbrev": area.abbreviation}
                                                     for area in paper.subject_areas]) \
                                              if paper.subject_areas else None

                    # Authors.
                    record["authors"] = tuple(
                        [{"id": str(author.auid) if author and str(author.auid) else None,
                        "name": "{} {}".format(author.given_name, author.surname) \
                                    if author and author.given_name and author.surname else
                                "{}".format(author.given_name) if author and author.given_name \
                                    and not author.surname else \
                                "{}".format(author.surname) if author and author.surname \
                                    and not author.given_name else None}
                        for author in paper.authors]) if paper.authors else None

                    record["author_affil"] = tuple(
                        [{"id": str(author.auid) if author and str(author.auid) else None,
                        "name": "{} {}".format(author.given_name, author.surname) \
                                    if author and author.given_name and author.surname else \
                                "{}".format(author.given_name) if author and author.given_name \
                                    and not author.surname else \
                                "{}".format(author.surname) if author and author.surname \
                                    and not author.given_name else None,
                        "affil_id": str(author.affiliation_id) if author and str(author.affiliation_id) else None,
                        "affiliation": author.organization if author and author.organization else None,
                        "country": author.country if author and author.country else None}
                        for author in paper.authorgroup]) if paper.authorgroup else None

                    # References.
                    record["ref_count"] = paper.refcount if paper.refcount else None
                    record["references"] = tuple([{"id": str(ref.id) if ref and str(ref.id) else None,
                                                "title": ref.title if ref and ref.title else None,
                                                "doi": ref.doi if ref and ref.doi else None,
                                                "authors": ref.authors if ref and ref.authors else None}
                                        for ref in paper.references]) if paper.references else None

                except (Scopus404Error, Scopus500Error, HTTPError, KeyError) as e:
                    record["id"] = str(key)
                    print(item["complete_name"], "-", key)
                    error = False
                except Scopus429Error as e:
                    # Removing the last item in _keys to assign it as new API key.
                    config["Authentication"]["APIKey"] = _keys.pop()
                    if len(_keys) == 0:
                        raise e
            data.append(record)
    return data

# Getting the list of manuscripts' EIDs for each members.
data_members = pd.read_csv("../data_analysis_pipeline/data/raw/members_stats.csv",
                              index_col=False)[["complete_name", "list_eids_documents"]]
data_members.list_eids_documents = data_members.list_eids_documents.apply(eval)
data_members = data_members.to_dict("records")

# Getting data.
papers = collect_data_manuscripts(data_members)

# Saving the data into CSV file.
pd.DataFrame(papers).to_csv("../data_analysis_pipeline/data/raw/manuscripts_group.csv",
                            index=False, quoting=csv.QUOTE_ALL)