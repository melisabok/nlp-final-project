# nlp-final-project
final project for the course NLP

### Data pipeline:

Download and save data in csv format

    arxiv-harvest.py


Process the harvested data and convert it into csv files:

    process-harvest.py -->

    data/corpus-abstracts.csv
    data/corpus-labels.csv
    data/corpus-titles.csv
    data/corpus-titles-abstracts.csv

Turn the whole corpus into a bag-of-words representation

    get-bow.py -->

    data/corpus-titles-abstracts.dict
    data/corpus-titles-abstracts.mm
    data/corpus-titles-abstracts.mm.index (not sure what this is for but it gets generated with the .mm file)


Build LSA and LDA models for the whole corpus

    get-models.py -->

    data/corpus-titles-abstracts.lsi
    data/corpus-titles-abstracts.lsi.projection
    data/corpus-titles-abstracts.lda
    data/corpus-titles-abstracts.lda.state

Get bag-of-words representations for individual categories (this will generate category bigrams as well)

    get-bow-by-category.py -c <category> -->

    data/<category>/corpus-titles-abstracts.dict
    data/<category>/corpus-titles-abstracts.mm
    data/<category>/corpus-titles-abstracts.mm.index (not sure what this is for but it gets generated with the .mm file)


Build LSA and LDA models by category

    get-models-by-category.py -c <category> -->

    data/<category>/corpus-titles-abstracts.lsi
    data/<category>/corpus-titles-abstracts.lsi.projection
    data/<category>/corpus-titles-abstracts.lda
    data/<category>/corpus-titles-abstracts.lda.state
