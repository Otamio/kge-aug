SUPPORTED_MODE = ['Quantile_Single', 'Quantile_Overlap', 'Quantile_Hierarchy',
                  'Fixed_Single', 'Fixed_Overlap', 'Fixed_Hierarchy', 'Kde_Single']

CHAINABLE_MODE = ['Quantile_Single', 'Quantile_Overlap', 'Quantile_Hierarchy',
                  'Fixed_Single', 'Fixed_Overlap', 'Fixed_Hierarchy', 'Kde_Single']

mapping_no_chain = {
    'Quantile_Single': 'QSN',
    'Quantile_Overlap': 'QON',
    'Quantile_Hierarchy': 'QHN',
    'Fixed_Single': 'FSN',
    'Fixed_Overlap': 'FON',
    'Fixed_Hierarchy': 'FHN',
    'Kde_Single': 'KSN'
}

mapping_chain = {
    'Quantile_Single': 'QSC',
    'Quantile_Overlap': 'QOC',
    'Quantile_Hierarchy': 'QHC',
    'Fixed_Single': 'FSC',
    'Fixed_Overlap': 'FOC',
    'Fixed_Hierarchy': 'FHC',
    'Kde_Single': 'KSC'
}
