##Archive categories
# cs.AR	Computer Science - Architecture
# cs.AI	Computer Science - Artificial Intelligence
# cs.CL	Computer Science - Computation and Language
# cs.CC	Computer Science - Computational Complexity
# cs.CE	Computer Science - Computational Engineering; Finance; and Science
# cs.CG	Computer Science - Computational Geometry
# cs.GT	Computer Science - Computer Science and Game Theory
# cs.CV	Computer Science - Computer Vision and Pattern Recognition
# cs.CY	Computer Science - Computers and Society
# cs.CR	Computer Science - Cryptography and Security
# cs.DS	Computer Science - Data Structures and Algorithms
# cs.DB	Computer Science - Databases
# cs.DL	Computer Science - Digital Libraries
# cs.DM	Computer Science - Discrete Mathematics
# cs.DC	Computer Science - Distributed; Parallel; and Cluster Computing
# cs.GL	Computer Science - General Literature
# cs.GR	Computer Science - Graphics
# cs.HC	Computer Science - Human-Computer Interaction
# cs.IR	Computer Science - Information Retrieval
# cs.IT	Computer Science - Information Theory
# cs.LG	Computer Science - Learning
# cs.LO	Computer Science - Logic in Computer Science
# cs.MS	Computer Science - Mathematical Software
# cs.MA	Computer Science - Multiagent Systems
# cs.MM	Computer Science - Multimedia
# cs.NI	Computer Science - Networking and Internet Architecture
# cs.NE	Computer Science - Neural and Evolutionary Computing
# cs.NA	Computer Science - Numerical Analysis
# cs.OS	Computer Science - Operating Systems
# cs.OH	Computer Science - Other
# cs.PF	Computer Science - Performance
# cs.PL	Computer Science - Programming Languages
# cs.RO	Computer Science - Robotics
# cs.SE	Computer Science - Software Engineering
# cs.SD	Computer Science - Sound
# cs.SC	Computer Science - Symbolic Computation

import time
import urllib2
import datetime
from itertools import ifilter
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import bibtexparser

OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"
YEAR = "2016"
def harvest(arxiv="cs"):
    df = pd.DataFrame(columns=("title", "abstract", "categories", "created", "id", "doi"))
    base_url = "http://export.arxiv.org/oai2?verb=ListRecords&"
    url = (base_url +
           "from="+YEAR+"-01-01&until="+YEAR+"-12-31&" +
           "metadataPrefix=arXiv&set=%s"%arxiv)
    
    while True:
        print "fetching", url
        try:
            response = urllib2.urlopen(url)
            
        except urllib2.HTTPError, e:
            if e.code == 503:
                to = int(e.hdrs.get("retry-after", 30))
                print "Got 503. Retrying after {0:d} seconds.".format(to)

                time.sleep(to)
                continue
                
            else:
                raise
            
        xml = response.read()

        root = ET.fromstring(xml)

        for record in root.find(OAI+'ListRecords').findall(OAI+"record"):
            arxiv_id = record.find(OAI+'header').find(OAI+'identifier')
            meta = record.find(OAI+'metadata')
            info = meta.find(ARXIV+"arXiv")
            created = info.find(ARXIV+"created").text
            created = datetime.datetime.strptime(created, "%Y-%m-%d")
            categories = info.find(ARXIV+"categories").text

            # if there is more than one DOI use the first one
            # often the second one (if it exists at all) refers
            # to an eratum or similar
            doi = info.find(ARXIV+"doi")
            if doi is not None:
                doi = doi.text.split()[0]
                
            contents = {'title': info.find(ARXIV+"title").text,
                        'id': info.find(ARXIV+"id").text,#arxiv_id.text[4:],
                        'abstract': info.find(ARXIV+"abstract").text.strip(),
                        'created': created,
                        'categories': categories.split(),
                        'doi': doi,
                        }

            df = df.append(contents, ignore_index=True)

        # The list of articles returned by the API comes in chunks of
        # 1000 articles. The presence of a resumptionToken tells us that
        # there is more to be fetched.
        token = root.find(OAI+'ListRecords').find(OAI+"resumptionToken")
        if token is None or token.text is None:
            break

        else:
            url = base_url + "resumptionToken=%s"%(token.text)
            
    return df

df = harvest()

df.to_csv('./data/papers_'+YEAR+'.csv', encoding = 'UTF-8')
