import argparse
from validphys.api import API
import numpy as np
from ruamel.yaml import YAML

yaml = YAML()

def main(config_path, output_path, factors):
    # Load the existing runcard
    with open(config_path, 'r') as file:
        runcard = yaml.load(file)

    x = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    y_2 = [0.023, -0.032, -0.005, 0.025, 0.051, 0.003, 0.0]
    y_T = [-0.319, -0.134, -0.052, 0.071, 0.030, 0.003, 0.0]
    y_L =  (y_2 - np.power(x, 0.05) * y_T)

    # Add filter rules to the runcard
    runcard['theorycovmatconfig']['H2_list'] = np.multiply(y_2, factors[0]).tolist()
    runcard['theorycovmatconfig']['HL_list'] = np.multiply(y_L, factors[1]).tolist()

    # Save the new runcard with added filter rules
    with open(output_path, 'w') as file:
        yaml.dump(runcard, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rescale the coefficient lists of the ABMP parametrisation for H_2 and H_L by user typed factors.")
    parser.add_argument('config_path', type=str, help='The path to the input runcard.')
    parser.add_argument('-o', '--output_path', type=str, default='runcard_with_.yml', help='The output path for the generated YAML file.')
    parser.add_argument('-p', '--parameter_factors', type=int, nargs=2, default=[1,1], help='The parameter factors that multiply the ABMP coefficients for H_2 and H_L, respectively.')

    args = parser.parse_args()
    main(args.config_path, args.output_path, args.parameter_factors)

