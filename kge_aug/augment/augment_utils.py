import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from bisect import bisect
from collections import defaultdict

SUPPORTED_MODE = ['Quantile', 'Jenks', 'Kde', 'QuantileDual', 'Hierarchy', 'Fixed', 'FixedDual', 'FixedHierarchy']
CHAINNABLE_MODE = ['Quantile', 'Jenks', 'Kde', 'QuantileDual', 'Hierarchy', 'Fixed', 'FixedDual', 'FixedHierarchy']


##########################################
###    Utility Functions
##########################################

def gen_qnode(property_, start, end, unit=None):
    if not unit:
        return f'Interval-{property_}({start}_{end})'
    return f'Interval-{property_}|{unit}({start}_{end})'


def gen_qlabel(property_, start, end, unit=None):
    if not unit:
        return f'{property_}({start}_{end})'
    return f'{property_}|{unit}({start}_{end})'


def gen_pnode(pnode, unit=None):
    if not unit:
        return f'Interval-{pnode}'
    return f'Interval-{pnode}|{unit}'


def gen_plabel(pnode, unit=None):
    if not unit:
        return pnode + ' (Interval)'
    return pnode + ' ' + unit + ' (Interval)'


def parse_number(s):
    '''
    Parsing the literals in Wikidata
    '''
    if s[0] != '-' and '-' in s:
        try:
            return int(s.split('-')[0])
        except:
            return None
            try:
                print(s)
                return int(s.split('-')[0][:-1]) * 10
            except:
                print(s)
                return int(s.split('-')[0][:-2]) * 100
    return float(s.split('^^')[0])


##########################################
###    Obtain data functions
##########################################

def get_data_lp(dataset):
    '''
    Get the entity file and literal file for Link Prediction
    '''
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
    '''
    Get the entity file and literal file for
    '''
    entities = pd.read_csv(f'datasets/{dataset}/numeric/train_kge', sep='\t', header=None)
    entities[0] = entities[0].apply(lambda x: x if not "org" in x else x.split("org")[1][:-1])
    entities[1] = entities[1].apply(lambda x: x if not "org" in x else x.split("org")[1][:-1])
    entities[2] = entities[2].apply(lambda x: x if not "org" in x else x.split("org")[1][:-1])
    entities.columns = ['node1', 'label', 'node2']

    values = pd.read_csv(f'datasets/{dataset}/numeric/train_100', sep='\t', header=None)
    values[0] = values[0].apply(lambda x: x if not "org" in x else x.split("org")[1][:-1])
    values[1] = values[1].apply(lambda x: x if not "com" in x else x.split("com")[1][:-1])
    values[1] = values[1].apply(lambda x: x if not "org" in x else x.split("org")[1][:-1])
    values.columns = ['node1', 'label', 'node2']
    values = values[values['node2'].notnull()]
    values = values.reset_index(drop=True)

    return entities, values


##########################################
###    Partition Functions
##########################################

def get_intervals_for_values_based_on_percentile(values, num_bins):
    values = np.array(values)
    indexes = np.arange(len(values))

    # sort values and corresponding labels in ascending order
    indexes = np.array([i for i, v in sorted(zip(indexes, values), key=lambda pair: pair[1])])
    values.sort()

    interval_bounds = []
    for i in range(num_bins):
        index_of_lbound = int(((i) / num_bins) * len(values))
        index_of_ubound = int(((i + 1) / num_bins) * len(values)) - 1
        lbound = values[index_of_lbound]
        ubound = values[index_of_ubound]
        interval_bounds.append((lbound, ubound))
    intervals_for_values = []
    cur_interval_idx = 0
    for i in range(len(values)):
        while values[i] > interval_bounds[cur_interval_idx][1]:
            cur_interval_idx += 1
        intervals_for_values.append(interval_bounds[cur_interval_idx])

    # rearrange intervals to original order of values
    intervals_for_values_unscrambled = np.zeros(len(intervals_for_values), dtype=tuple)
    for i in range(len(indexes)):
        intervals_for_values_unscrambled[indexes[i]] = intervals_for_values[i]

    return intervals_for_values_unscrambled


def get_bins_jenks(df):
    '''
    MODE: JENKS
    Partition the 1D array in Jenks Natural Breaks Algorithms
    '''
    import jenkspy

    num_bins = len(get_bins_kde(df))
    values = np.array(df.loc[:, "node2"])
    if num_bins < 2:
        return [min(values)]

    breaks = jenkspy.jenks_breaks(values, num_bins)
    return breaks[:-1]


def get_bins_kde(df):
    '''
    MODE: KDE
    Partition the 1D array in KDE
    '''
    from sklearn.neighbors import KernelDensity
    from scipy.signal import argrelextrema
    from matplotlib.pyplot import plot

    values = np.array(df.loc[:, "node2"])
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(values.reshape(-1, 1))
    s = np.linspace(min(values), max(values))
    e = kde.score_samples(s.reshape(-1, 1))
    plot(s, e)
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    return [min(values)] + [s[i] for i in mi]


def get_bins_quantile(df, num_bins=None):
    '''
    MODE: ORIGIN, QUANTILE
    Partition the 1D array into a list of entity nodes
    '''
    if num_bins is None:
        num_bins = len(get_bins_kde(df))

    # Add two columns
    try:
        df.insert(loc=len(df.columns), column="lower_bound", value=["" for i in range(df.shape[0])])
        df.insert(loc=len(df.columns), column="upper_bound", value=["" for i in range(df.shape[0])])
    except:
        pass

    values = np.array(df.loc[:, "node2"])

    # Partition the nodes based on node2
    intervals_for_values = get_intervals_for_values_based_on_percentile(values, num_bins)

    intervals = set(intervals_for_values)
    set_ = set()

    for i, (p1, p2) in enumerate(sorted(intervals, key=lambda x: x[0])):
        set_.add(p1)
    return sorted(set_)


def get_bins_fixed_length(df, num_bins=None):
    '''
    MODE: FIXED_LENGTH
    '''
    if num_bins is None:
        num_bins = len(get_bins_kde(df))
    values = np.array(df.loc[:, "node2"])
    return np.linspace(values.min(), values.max() - values.min(), num_bins + 1)[:-1]


##########################################
###    Edge Generation Utility Functions
##########################################

def get_edge_starts(df_sli, MODE, num_bins=None):
    '''
    Get the start of each bin
    '''
    if MODE == 'Quantile':
        bins = get_bins_quantile(df_sli, num_bins)
    elif MODE == 'Fixed':
        bins = get_bins_fixed_length(df_sli, num_bins)
    elif MODE == 'Jenks':
        bins = get_bins_jenks(df_sli)
    elif MODE == 'Kde':
        bins = get_bins_kde(df_sli)
    elif MODE == 'QuantileDual':
        bins = get_bins_quantile(df_sli, num_bins)
    else:  # Unsupported Mode
        return None
    return bins


def create_chain(qnodes_collect, property_):
    '''
    create the chain needed
    '''
    qnode_chain = []
    for i in range(len(qnodes_collect)):
        if i > 0:
            qnode_chain.append({
                'node1': qnodes_collect[i],
                'label': property_ + '_prev',
                'node2': qnodes_collect[i - 1]
            })
        if i < len(qnodes_collect) - 1:
            qnode_chain.append({
                'node1': qnodes_collect[i],
                'label': property_ + '_next',
                'node2': qnodes_collect[i + 1]
            })
    return qnode_chain


def create_hierarchy(qnodes_collect_A, qnodes_collect_B, property_):
    '''
    create the hierarchy needed
    '''
    qnode_hierarchy = []
    for i in range(len(qnodes_collect_A)):
        qnode_hierarchy.append({
            'node1': qnodes_collect_A[i],
            'label': property_ + '_is_subgroup_of',
            'node2': qnodes_collect_B[i // 2]
        })
        qnode_hierarchy.append({
            'node1': qnodes_collect_B[i // 2],
            'label': property_ + '_has_subgroup',
            'node2': qnodes_collect_A[i]
        })
    return qnode_hierarchy


def create_literal_labels(bins, property_):
    '''
    Create labels for quantity_intervals
    '''
    qnodes_collect = []
    qnodes_label_edges = []

    for i in range(len(bins)):
        if i == 0:
            start = bins[i]
        else:
            start = bins[i] + 1e-6

        if i == len(bins) - 1:
            end = 'inf'
        else:
            end = bins[i + 1]

        _qnode = gen_qnode(property_, start, end)
        _qlabel = gen_qlabel(property_, start, end)

        qnodes_collect.append(_qnode)

        qnodes_label_edges.append({
            'node1': _qnode,
            'label': 'label',
            'node2': _qlabel
        })
    return qnodes_collect, qnodes_label_edges


def create_property_labels(property_, unit):
    ''' Generate the pnodes '''
    pnodes_collect = []
    pnodes_edges = []
    pnodes_label_edges = []

    _pnode = gen_pnode(property_, unit)
    pnodes_collect.append(_pnode)
    _plabel = gen_plabel(property_, unit)
    pnodes_label_edges.append({
        'node1': _pnode,
        'label': 'label',
        'node2': _plabel
    })
    return pnodes_collect, pnodes_edges, pnodes_label_edges


def create_numeric_edges(df_sli, bins, qnodes_collect):
    ''' Create numeric edges '''
    numeric_edges = []
    if len(qnodes_collect) == 0:
        return numeric_edges

    for i, row in df_sli.iterrows():
        _pnode = gen_pnode(row['label'])
        try:
            _qnode = qnodes_collect[bisect(bins, row['node2']) - 1]
        except:
            _qnode = qnodes_collect[0]
        numeric_edges.append({
            'node1': row['node1'],
            'label': _pnode,
            'node2': _qnode
        })
    return numeric_edges


##########################################
###    Edge Generation Functions
##########################################

def generate_edges(df_sli, property_, MODE, num_bins=None, unit=None):
    '''
    Generate a 1D partition of literal nodes and add them to entity nodes
    '''
    bins = get_edge_starts(df_sli, MODE, num_bins)

    # print(property_, bins)
    # Generate the qnode
    qnodes_collect, qnodes_label_edges = create_literal_labels(bins, property_)

    # Connect with quantity nodes
    qnode_chain = create_chain(qnodes_collect, property_)
    # Generate the pnodes
    pnodes_collect, pnodes_edges, pnodes_label_edges = create_property_labels(property_, unit)
    # Generate numeric edges
    numeric_edges = create_numeric_edges(df_sli, bins, qnodes_collect)

    return qnode_chain, qnodes_label_edges, pnodes_edges, pnodes_label_edges, numeric_edges


def generate_edges_dualLink(df_sli, property_, num_bins, unit=None, mode='QuantileDual'):
    '''
    One numeric edge = 2 links
    '''
    if 'Quantile' in mode:
        bs = get_edge_starts(df_sli, 'QuantileDual', num_bins * 2)
    else:
        bs = get_edge_starts(df_sli, 'Fixed', num_bins * 2)
    bin_A, bin_B = bs[0::2], bs[1::2][:-1]

    qnodes_collect_A, qnodes_label_edges_A = create_literal_labels(bin_A, property_)
    qnodes_collect_B, qnodes_label_edges_B = create_literal_labels(bin_B, property_)

    qnodes_collect_all = []
    for i in range(len(qnodes_collect_B)):
        qnodes_collect_all.append(qnodes_collect_A[i])
        qnodes_collect_all.append(qnodes_collect_B[i])
    if len(qnodes_collect_A) > len(qnodes_collect_B):
        qnodes_collect_all += qnodes_collect_A[len(qnodes_collect_B):]

    qnode_chain = create_chain(qnodes_collect_all, property_)

    pnodes_collect, pnodes_edges, pnodes_label_edges = create_property_labels(property_, unit)

    numeric_edges_A = create_numeric_edges(df_sli, bin_A, qnodes_collect_A)
    numeric_edges_B = create_numeric_edges(df_sli, bin_B, qnodes_collect_B)

    # qnode_chain = qnode_chain_A + qnode_chain_B
    qnodes_label_edges = qnodes_label_edges_A + qnodes_label_edges_B
    numeric_edges = numeric_edges_A + numeric_edges_B

    return qnode_chain, qnodes_label_edges, pnodes_edges, pnodes_label_edges, numeric_edges


def generate_edges_hierarchy(df_sli, property_, levels=3, unit=None, mode='Hierarchy'):
    '''
    Link Hierarchy
    '''
    from functools import reduce

    if "Fixed" in mode:
        bs = get_edge_starts(df_sli, 'Fixed', 2 ** levels)
    else:
        bs = get_edge_starts(df_sli, 'Quantile', 2 ** levels)
    bs_list = []
    for l in range(levels + 1):
        bs_list.append(bs[0::2 ** l])

    qnodes_collect_list, qnodes_label_edges_list = list(), list()
    for l in range(levels + 1):
        _a, _b = create_literal_labels(bs_list[l], property_)
        qnodes_collect_list.append(_a)
        qnodes_label_edges_list.append(_b)

    qnode_chain_list = list()
    for l in range(levels + 1):
        qnode_chain_list.append(create_chain(qnodes_collect_list[l], property_))
    for l in range(levels - 1):
        qnode_chain_list.append(create_hierarchy(qnodes_collect_list[l],
                                                 qnodes_collect_list[l + 1],
                                                 property_))

    pnodes_collect, pnodes_edges, pnodes_label_edges = create_property_labels(property_, unit)

    numeric_edges_list = list()
    for l in range(levels + 1):
        numeric_edges_list.append(create_numeric_edges(df_sli, bs_list[l], qnodes_collect_list[l]))

    qnode_chain = reduce(lambda x, y: x + y, qnode_chain_list)
    qnodes_label_edges = reduce(lambda x, y: x + y, qnodes_label_edges_list)
    numeric_edges = reduce(lambda x, y: x + y, numeric_edges_list)

    return qnode_chain, qnodes_label_edges, pnodes_edges, pnodes_label_edges, numeric_edges


##########################################
###    Edge Creation Functions
##########################################

def create_new_edges(df, MODE, num_bins=None, num_levels=None):
    '''
    Create the new edges based on the partitioned data
    '''
    qnodes_edges = []
    qnodes_label_edges = []
    pnodes_edges = []
    pnodes_label_edges = []
    numeric_edges = []
    numeric_edges_raw = None

    for property_ in tqdm(df['label'].unique()):

        try:
            sli = df[df['label'] == property_]
            if len(sli) < 100:
                continue

            if MODE == "QuantileDual" or MODE == "FixedDual":
                assert (not num_bins is None)
                a, b, c, d, e = generate_edges_dualLink(sli, property_, num_bins, MODE)
            elif MODE == "Hierarchy" or MODE == "FixedHierarchy":
                a, b, c, d, e = generate_edges_hierarchy(sli, property_, num_levels, MODE)
            else:
                a, b, c, d, e = generate_edges(sli, property_, MODE, num_bins)

            qnodes_edges += a
            qnodes_label_edges += b
            pnodes_edges += c
            pnodes_label_edges += d
            numeric_edges += e
            if numeric_edges_raw is None:
                numeric_edges_raw = sli
            else:
                numeric_edges_raw = pd.concat([numeric_edges_raw, sli])
        except Exception as e:
            print(f"Error encountered at property {property_}. Size {len(sli)}. Continue...")
            import traceback
            traceback.print_exc()

    numeric_edges_processed = pd.DataFrame(numeric_edges)

    return numeric_edges_processed, numeric_edges_raw, qnodes_edges


##########################################
###    Main Function
##########################################

def augment_lp(entities, df, dataset, mode, bins=None, levels=3):
    if mode in CHAINNABLE_MODE:
        print(f'Running mode {mode}')

        numeric_edges_processed, _, qnode_edges = create_new_edges(df, mode, bins, levels)
        # Write the original version
        pd.concat([entities, numeric_edges_processed]).to_csv(f'datasets/{dataset}/data/train_{mode}.tsv',
                                                              sep='\t', header=None, index=None)
        # Write the chainning version
        pd.concat([entities, numeric_edges_processed,
                   pd.DataFrame(qnode_edges)]).to_csv(f'datasets/{dataset}/data/train_{mode}_Chain.tsv',
                                                      sep='\t', header=None, index=None)


def augment_np(entities, values, dataset, mode, bins=None, levels=3):
    if mode in CHAINNABLE_MODE:

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

        import json
        try:
            os.mkdir(f'datasets/{dataset}/stats')
        except:
            pass
        with open(f'datasets/{dataset}/stats/train_{mode}.json', 'w+') as fd:
            json.dump(medians_dict, fd, indent=2)

        try:
            os.mkdir(f'datasets/{dataset}/processed')
        except:
            pass

        # Write the original version
        pd.concat([entities, numeric_edges_processed]).to_csv(f'datasets/{dataset}/numeric/train_{mode}.tsv',
                                                              sep='\t', header=None, index=None)

        # Write the chaining version
        pd.concat([entities, numeric_edges_processed,
                   pd.DataFrame(qnode_edges)]).to_csv(f'datasets/{dataset}/numeric/train_{mode}_Chain.tsv',
                                                      sep='\t', header=None, index=None)
