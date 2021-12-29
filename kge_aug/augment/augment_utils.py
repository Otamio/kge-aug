from tqdm import tqdm
from bisect import bisect
from utils import *
from loader import *
from partition import *


##########################################
#    Edge Generation Utility Functions
##########################################

def get_edge_starts(df_sli, mode, num_bins=None):
    """
    Get the start of each bin
    """
    if mode.startswith('Quantile'):
        bins = get_bins_quantile(df_sli, num_bins)
    elif mode.startswith('Fixed'):
        bins = get_bins_fixed_length(df_sli, num_bins)
    elif mode.startswith('Jenks'):
        bins = get_bins_jenks(df_sli)
    elif mode.startswith('Kde'):
        bins = get_bins_kde(df_sli)
    else:  # Unsupported Mode
        return None
    return bins


def create_chain(qnodes_collect, property_, reverse_chain=True):
    """
    create the chain needed
    """
    qnode_chain = []
    for i in range(len(qnodes_collect)):
        if reverse_chain and i > 0:
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


def create_hierarchy(qnodes_collect_a, qnodes_collect_b, property_, reverse_relation=True):
    """
    create the hierarchy needed
    """
    qnode_hierarchy = []
    for i in range(len(qnodes_collect_a)):
        qnode_hierarchy.append({
            'node1': qnodes_collect_a[i],
            'label': property_ + '_is_subgroup_of',
            'node2': qnodes_collect_b[i // 2]
        })
        if reverse_relation:
            qnode_hierarchy.append({
                'node1': qnodes_collect_b[i // 2],
                'label': property_ + '_has_subgroup',
                'node2': qnodes_collect_a[i]
            })
    return qnode_hierarchy


def create_literal_labels(bins, property_):
    """
    Create labels for quantity_intervals
    """
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
    """ Generate the pnodes """
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


def create_numeric_edges(df_sli, bins, qnodes_collect, suffix=""):
    """ Create numeric edges """
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
            'label': f"{_pnode}{suffix}",
            'node2': _qnode
        })
    return numeric_edges


##########################################
#    Edge Generation Functions
##########################################

def generate_edges_single(df_sli, property_, num_bins=None, unit=None, mode='Quantile_Single'):
    '''
    Generate a 1D partition of literal nodes and add them to entity nodes
    '''
    bins = get_edge_starts(df_sli, mode, num_bins)

    # Generate the qnode
    qnodes_collect, qnodes_label_edges = create_literal_labels(bins, property_)

    # Connect with quantity nodes
    qnode_chain = create_chain(qnodes_collect, property_, reverse_chain=False)
    # Generate the pnodes
    pnodes_collect, pnodes_edges, pnodes_label_edges = create_property_labels(property_, unit)
    # Generate numeric edges
    numeric_edges = create_numeric_edges(df_sli, bins, qnodes_collect)

    return qnode_chain, qnodes_label_edges, pnodes_edges, pnodes_label_edges, numeric_edges


def generate_edges_overlapping(df_sli, property_, num_bins=None, unit=None, mode='Quantile_Overlap'):
    '''
    One numeric edge = 2 links
    '''
    bs = get_edge_starts(df_sli, mode, num_bins)
    # print(bs)
    bins_a, bins_b = bs[0::2], [bs[0]] + bs[1::2][:-1]
    if len(bins_a) == len(bins_b):
        bins_b.append(bs[-1])
    else:
        bins_a.append(bs[-1])
    # print(bins_a, bins_b)

    qnodes_collect_a, qnodes_label_edges_a = create_literal_labels(bins_a, property_)
    qnodes_collect_b, qnodes_label_edges_b = create_literal_labels(bins_b, property_)

    qnodes_collect_all = []
    for i in range(len(qnodes_collect_a)):
        qnodes_collect_all.append(qnodes_collect_a[i])
        qnodes_collect_all.append(qnodes_collect_b[i])
    if len(qnodes_collect_a) > len(qnodes_collect_b):
        qnodes_collect_all += qnodes_collect_a[len(qnodes_collect_b):]

    qnode_chain = create_chain(qnodes_collect_all, property_, reverse_chain=False)

    pnodes_collect, pnodes_edges, pnodes_label_edges = create_property_labels(property_, unit)

    numeric_edges_a = create_numeric_edges(df_sli, bins_a, qnodes_collect_a, suffix="_left")
    numeric_edges_b = create_numeric_edges(df_sli, bins_b, qnodes_collect_b, suffix="_right")

    # qnode_chain = qnode_chain_a + qnode_chain_a
    qnodes_label_edges = qnodes_label_edges_a + qnodes_label_edges_b
    numeric_edges = numeric_edges_a + numeric_edges_b

    return qnode_chain, qnodes_label_edges, pnodes_edges, pnodes_label_edges, numeric_edges


def generate_edges_hierarchy(df_sli, property_, levels=3, unit=None, mode='Quantile_Hierarchy'):
    """
    Link Hierarchy
    """
    from functools import reduce

    bs = get_edge_starts(df_sli, mode, 2 ** levels)
    bs_list = []
    for lv in range(levels + 1):
        bs_list.append(bs[0::2 ** lv])

    qnodes_collect_list, qnodes_label_edges_list = list(), list()
    for lv in range(levels + 1):
        _a, _b = create_literal_labels(bs_list[lv], property_)
        qnodes_collect_list.append(_a)
        qnodes_label_edges_list.append(_b)

    qnode_chain_list = list()
    for lv in range(levels + 1):
        qnode_chain_list.append(create_chain(qnodes_collect_list[lv], property_, reverse_chain=False))
    for lv in range(levels - 1):
        qnode_chain_list.append(create_hierarchy(qnodes_collect_list[lv],
                                                 qnodes_collect_list[lv + 1],
                                                 property_, reverse_relation=False))

    pnodes_collect, pnodes_edges, pnodes_label_edges = create_property_labels(property_, unit)

    numeric_edges_list = list()
    for l in range(levels + 1):
        numeric_edges_list.append(create_numeric_edges(df_sli, bs_list[l], qnodes_collect_list[l]))

    qnode_chain = reduce(lambda x, y: x + y, qnode_chain_list)
    qnodes_label_edges = reduce(lambda x, y: x + y, qnodes_label_edges_list)
    numeric_edges = reduce(lambda x, y: x + y, numeric_edges_list)

    return qnode_chain, qnodes_label_edges, pnodes_edges, pnodes_label_edges, numeric_edges


##########################################
#    Edge Creation Functions
##########################################

def create_new_edges(df, mode, num_bins=None, num_levels=None):
    """
    Create the new edges based on the partitioned data
    """
    qnodes_edges = []
    qnodes_label_edges = []  # (metadata) entity labels
    pnodes_edges = []
    pnodes_label_edges = []  # (metadata) property labels
    numeric_edges = []
    numeric_edges_raw = None  # numeric edges (node2 as numbers)

    for property_ in tqdm(df['label'].unique()):

        # Iterate through each numeric property
        sli = df[df['label'] == property_]
        if len(sli) < 100:  # Filter out rare properties
            continue

        try:
            if mode.endswith("Single"):
                assert(num_bins is not None)
                a, b, c, d, e = generate_edges_single(sli, property_, num_bins=num_bins, unit=None, mode=mode)
            elif mode.endswith("Overlap"):
                assert(num_bins is not None)
                a, b, c, d, e = generate_edges_overlapping(sli, property_, num_bins=num_bins, unit=None, mode=mode)
            elif mode.endswith("Hierarchy"):
                assert(num_levels is not None)
                a, b, c, d, e = generate_edges_hierarchy(sli, property_, levels=num_levels, unit=None, mode=mode)
            else:
                print("Unsupported data type!")
                continue

            qnodes_edges += a
            qnodes_label_edges += b
            pnodes_edges += c
            pnodes_label_edges += d
            numeric_edges += e
            numeric_edges_raw = sli if numeric_edges_raw is None else pd.concat([numeric_edges_raw, sli])

        except TypeError as e:
            assert(sli is not None)
            print(f"Error encountered at property {property_}. Size {len(sli)}. Error: {e}. Continue...")
            import traceback
            traceback.print_exc()

    numeric_edges_processed = pd.DataFrame(numeric_edges)

    return numeric_edges_processed, numeric_edges_raw, qnodes_edges
