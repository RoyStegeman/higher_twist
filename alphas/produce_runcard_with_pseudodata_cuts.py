import argparse
from validphys.api import API
import numpy as np
from ruamel.yaml import YAML

yaml = YAML()

def main(config_path, output_path):
    # Load the existing runcard
    with open(config_path, 'r') as file:
        runcard = yaml.load(file)

    config = dict(
        dataset_inputs=runcard['dataset_inputs'],
        theoryid=runcard['theory']['theoryid'],
        use_cuts='internal',
        **runcard['theorycovmatconfig'],
    )

    # Get central values of dataset inputs
    loaded_dataset_inputs = API.groups_dataset_inputs_loaded_cd_with_cuts(**config)
    central_values = np.concatenate([cd.central_values for cd in loaded_dataset_inputs])

    # Get uncertainties from the covariance matrix
    covmat = API.dataset_inputs_sampling_covmat(**config)
    uncs = np.sqrt(np.diag(covmat))

    # Determine indices of data points to cut based on central values and uncertainties
    mask = (central_values - 2 * uncs) < 0
    datapoint_index = API.groups_index(**config)
    datapoints_to_cut = datapoint_index[mask]

    import ipdb; ipdb.set_trace()

    # Add filter rules to the runcard
    runcard['added_filter_rules'] = []
    for dp in datapoints_to_cut:
        runcard['added_filter_rules'].append(dict(dataset=dp[1], rule=f'idat != {dp[2]}'))

    # Add inline comment to separate the added filter rules
    runcard.yaml_set_comment_before_after_key('added_filter_rules', before="\n############################################################")

    # Save the new runcard with added filter rules
    with open(output_path, 'w') as file:
        yaml.dump(runcard, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add filter rules to an existing n3fit runcard to remove all datapoints within 2 std from 0.")
    parser.add_argument('config_path', type=str, help='The path to the input runcard.')
    parser.add_argument('-o', '--output_path', type=str, default='runcard_with_filterrules.yml', help='The output path for the generated YAML file.')

    args = parser.parse_args()
    main(args.config_path, args.output_path)

