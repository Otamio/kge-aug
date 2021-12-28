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
    if mode == 'Quantile':
        bins = get_bins_quantile(df_sli, num_bins)
    elif mode == 'Fixed':
        bins = get_bins_fixed_length(df_sli, num_bins)
    elif mode == 'Jenks':
        bins = get_bins_jenks(df_sli)
    elif mode == 'Kde':
        bins = get_bins_kde(df_sli)
    elif mode == 'QuantileDual':
        bins = get_bins_quantile(df_sli, num_bins)
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


def create_hierarchy(qnodes_collect_A, qnodes_collect_B, property_):
    """
    create the hierarchy needed
    """
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


def create_numeric_edges(df_sli, bins, qnodes_collect):
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
            'label': _pnode,
            'node2': _qnode
        })
    return numeric_edges


##########################################
#    Edge Generation Functions
##########################################

def generate_edges(df_sli, property_, mode, num_bins=None, unit=None):
    '''
    Generate a 1D partition of literal nodes and add them to entity nodes
    '''
    bins = get_edge_starts(df_sli, mode, num_bins)

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
#    Edge Creation Functions
##########################################

def create_new_edges(df, mode, num_bins=None, num_levels=None):
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

            if mode == "QuantileDual" or mode == "FixedDual":
                assert (not num_bins is None)
                a, b, c, d, e = generate_edges_dualLink(sli, property_, num_bins, mode)
            elif mode == "Hierarchy" or mode == "FixedHierarchy":
                a, b, c, d, e = generate_edges_hierarchy(sli, property_, num_levels, mode)
            else:
                a, b, c, d, e = generate_edges(sli, property_, mode, num_bins)

            qnodes_edges += a
            qnodes_label_edges += b
            pnodes_edges += c
            pnodes_label_edges += d
            numeric_edges += e
            numeric_edges_raw = sli if numeric_edges_raw is None else pd.concat([numeric_edges_raw, sli])

        except Exception as e:
            print(f"Error encountered at property {property_}. Size {len(sli)}. Continue...")
            import traceback
            traceback.print_exc()

    numeric_edges_processed = pd.DataFrame(numeric_edges)

    return numeric_edges_processed, numeric_edges_raw, qnodes_edges
