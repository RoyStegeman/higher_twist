from validphys.api import API
import numpy as np
from ruamel.yaml import YAML

yaml = YAML()

fitname = "240519-01-rs-nnpdf40-alphas-tcm-mhou"
output_path = "runcard_with_filterrules.yml"

config = dict(
    fit = fitname,
    dataset_inputs = {'from_': 'fit'},
    use_cuts='fromfit',
    theoryid= {'from_': 'theory'},
    theory= {'from_': 'fit'},
)

# get central values
loaded_dataset_inputs = API.groups_dataset_inputs_loaded_cd_with_cuts(**config)
central_values = np.concatenate( [cd.central_values for cd in loaded_dataset_inputs] )

# get unceratinaties
covmat = API.dataset_inputs_sampling_covmat(**config)
uncs = np.sqrt(np.diag(covmat))

# get dataset and id of points to cut
mask = (central_values - 2*uncs) > 0
datapoint_index = API.groups_index(**config)
datapoints_to_cut = datapoint_index[mask]

# load yaml file of input runcard
input_yaml_path=API.fit(**config).path / 'filter.yml'
with open(input_yaml_path, 'r') as file:
    output_yaml = yaml.load(file)

# add cuts info to runcard
output_yaml['added_filter_rules'] = []
for dp in datapoints_to_cut:
    output_yaml['added_filter_rules'].append(dict(dataset=dp[1], rule=f'idat != {dp[2]}'))

# add inline comment to separete the added_filter_rules
output_yaml.yaml_set_comment_before_after_key('added_filter_rules', before="\n############################################################")
# safe new output runcard
with open(output_path, 'w') as file:
    yaml.dump(output_yaml, file)

