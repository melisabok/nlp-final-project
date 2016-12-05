import csv
import os

# Overwrite the data files
with open('./data/corpus-abstracts.csv','wb') as x, open('./data/corpus-labels.csv','wb') as y, open('./data/corpus-titles.csv','wb') as z:
    pass

with open('./data/corpus-abstracts.csv','ab') as x, open('./data/corpus-labels.csv','ab') as y, open('./data/corpus-titles.csv','ab') as z, open('./data/corpus-titles-abstracts.csv','ab') as a:
    for path in ['./data/papers_'+year+'.csv' for year in ['2012','2013','2014','2015','2016']]:
        with open(path) as f:
            reader = csv.reader(f)
            abstracts_writer = csv.writer(x,delimiter='\n')
            labels_writer = csv.writer(y,delimiter='\n')
            titles_writer = csv.writer(z,delimiter='\n')
            titles_abstracts_writer = csv.writer(a,delimiter='\n')

            for paper in reader:
                title = paper[1].replace('\n',' ')
                abstract = paper[2].replace('\n',' ')
                label = paper[3]
                label = label.replace('[','')
                label = label.replace(']','')
                label = label.replace(' ','')
                titles_writer.writerow([title])
                abstracts_writer.writerow([abstract])
                labels_writer.writerow([label])
                titles_abstracts_writer.writerow([title+' '+abstract])
