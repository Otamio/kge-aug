import pandas as pd

##########################################
#    Obtain data functions
##########################################


def get_data_lp(dataset):
    """
    Get the entity file and literal file for Link Prediction
    """
    if dataset == "WikidataDWD":
        entities = pd.read_csv(f'datasets/{dataset}/data/train.tsv', sep='\t', usecols=[0, 1, 2])
    else:
        entities = pd.read_csv(f'datasets/{dataset}/data/train.tsv', sep='\t', header=None, usecols=[0, 1, 2])
    entities.columns = ['node1', 'label', 'node2']

    df = pd.read_csv(f'datasets/{dataset}/data/numerical_literals.tsv', sep='\t', header=None, usecols=[0, 1, 2])
    df.columns = ['node1', 'label', 'node2']
    df = df[df['node2'].notnull()]
    df = df.reset_index(drop=True)

    return entities, df


def get_data_np(dataset):
    """
    Get the entity file and literal file for
    """
    entities = pd.read_csv(f'datasets/{dataset}/numeric/train_kge', sep='\t', header=None)
    entities[0] = entities[0].apply(lambda x: x if "org" not in x else x.split("org")[1][:-1])
    entities[1] = entities[1].apply(lambda x: x if "org" not in x else x.split("org")[1][:-1])
    entities[2] = entities[2].apply(lambda x: x if "org" not in x else x.split("org")[1][:-1])
    entities.columns = ['node1', 'label', 'node2']

    values = pd.read_csv(f'datasets/{dataset}/numeric/train_100', sep='\t', header=None)
    values[0] = values[0].apply(lambda x: x if "org" not in x else x.split("org")[1][:-1])
    values[1] = values[1].apply(lambda x: x if "com" not in x else x.split("com")[1][:-1])
    values[1] = values[1].apply(lambda x: x if "org" not in x else x.split("org")[1][:-1])
    values.columns = ['node1', 'label', 'node2']
    values = values[values['node2'].notnull()]
    values = values.reset_index(drop=True)

    return entities, values
