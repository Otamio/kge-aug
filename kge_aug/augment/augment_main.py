from constant import *
from augment_utils import *
from collections import defaultdict
import os
import json


##########################################
#    Main Function
##########################################

def augment_lp(entities, df, dataset, mode, bins=None, levels=3):

    suffix = bins if bins is not None else levels

    if mode in CHAINABLE_MODE:

        print(f'Running mode {mode}')

        numeric_edges_processed, _, qnode_edges = create_new_edges(df, mode, bins, levels)

        # Write the augmented version (without chain)
        pd.concat([entities, numeric_edges_processed])\
            .to_csv(f'datasets/{dataset}/processed/train_{mapping_no_chain[mode]}_{suffix}.txt',
                    sep='\t', header=False, index=False)
        # Write the augmented version (with chain)
        pd.concat([entities, numeric_edges_processed, pd.DataFrame(qnode_edges)])\
            .to_csv(f'datasets/{dataset}/processed/train_{mapping_chain[mode]}_{suffix}.txt',
                    sep='\t', header=False, index=False)


def augment_np(entities, values, dataset, mode, bins=None, levels=3):

    suffix = bins if bins is not None else levels

    if mode in CHAINABLE_MODE:

        print(f'Running mode {mode}')

        numeric_edges_processed, numeric_edges_raw, qnode_edges = create_new_edges(values, mode, bins, levels)

        medians_dict = {}
        collections = defaultdict(list)
        collections_raw = defaultdict(list)

        for i, row in numeric_edges_raw.iterrows():
            collections_raw[row['node1'] + '__' + row['label']].append(row['node2'])

        for i, row in numeric_edges_processed.iterrows():
            key = row['node1'] + '__' + row['label'].split('-')[1]
            for item in collections_raw[key]:
                collections[row['node2']].append(item)

        for k, v in collections.items():
            medians_dict[k] = np.median(v)

        # Finally, add the median of each property as a baseline
        for property_ in numeric_edges_raw['label'].unique():
            medians_dict[property_] = numeric_edges_raw[numeric_edges_raw['label'] == property_]['node2'].median()

        try:
            os.mkdir(f'datasets/{dataset}/stats')
        except FileExistsError:
            pass
        with open(f'datasets/{dataset}/stats/train_{mapping_no_chain[mode]}.json', 'w+') as fd:
            json.dump(medians_dict, fd, indent=2)
        with open(f'datasets/{dataset}/stats/train_{mapping_chain[mode]}.json', 'w+') as fd:
            json.dump(medians_dict, fd, indent=2)

        try:
            os.mkdir(f'datasets/{dataset}/processed')
        except FileExistsError:
            pass

        # Write the original version
        pd.concat([entities, numeric_edges_processed])\
            .to_csv(f'datasets/{dataset}/numeric/train_{mapping_no_chain[mode]}_{suffix}.tsv',
                    sep='\t', header=False, index=False)

        # Write the chaining version
        pd.concat([entities, numeric_edges_processed, pd.DataFrame(qnode_edges)])\
            .to_csv(f'datasets/{dataset}/numeric/train_{mapping_chain[mode]}_{suffix}.tsv',
                    sep='\t', header=False, index=False)
