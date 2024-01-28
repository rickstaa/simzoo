# Stable-gym Translation Validation

This directory contains scripts to validate the translations of Gym environments in the [stable-gym](https://github.com/rickstaa/stable-gym/tree/han2020) package. These scripts are designed to compare the translated environments against their original versions as found in the [Actor-critic-with-stability-guarantee](https://github.com/rickstaa/Actor-critic-with-stability-guarantee/tree/rstaa2024) repository.

Given the ageing codebase of the original environments and their reliance on older versions of Gymnasium (originally OpenAI Gym) and numpy libraries, unit tests for deterministic behaviour need to be revised due to differences in seeding implementations. This necessitates manual validation processes, as described below, to ensure accuracy.

> \[!NOTE]\
> The scripts in this directory are intended to work in conjunction with the corresponding scripts in the [validation](https://github.com/rickstaa/Actor-critic-with-stability-guarantee/tree/rstaa2024/validation) directory, located on the [rstaa2024](https://github.com/rickstaa/Actor-critic-with-stability-guarantee/tree/rstaa2024) branch of the [Actor-critic-with-stability-guarantee](https://github.com/rickstaa/Actor-critic-with-stability-guarantee/tree/rstaa2024) repository.

## Validation Procedure

### Environment Step Validation

To validate the step method of the environments:

1. **Install Dependencies**: First, install the required packages listed in the `requirements.txt` file found in the `validation` folders of both the `stable-gym` and `Actor-critic-with-stability-guarantee` repositories.

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Reference CSV**: Run the validation script in the `validation` folder in the [Actor-critic-with-stability-guarantee](https://github.com/rickstaa/Actor-critic-with-stability-guarantee/tree/rstaa2024/validation) repository to generate a reference CSV file. The CSV file will be saved in the `validation/results` folder of the repository.

   ```bash
   python original_repo_cartPoleCost_translation_validation.py
   ```

3. **Copy Validation CSV**: Transfer the generated reference CSV file to the `tests/manual/validation/results` folder of the [stable-gym](https://github.com/rickstaa/stable-gym/tree/han2020/tests/manual/validation/results) package.

4. **Run Validation Script**: Execute the corresponding validation script in the `tests/manual/validation` folder of the [stable-gym](https://github.com/rickstaa/stable-gym/tree/fix_tests/tests/manual/validation) package.

   ```bash
   python my_cartPoleCost_translation_validation.py
   ```

5. **Check Results**: The script compares the step output with the reference CSV. The translation is deemed accurate if the results match within a specified tolerance. Otherwise, further investigation is needed.

### Environment Reset Validation

For the reset method validation, due to differences in seeding methods, temporarily substitute the seeding method in the `stable-gym` environments with the [original seeding method from gym v0.12.1](https://github.com/openai/gym/blob/dbab98c367614dc7746d1a7e277de1beff9aa6b3/gym/utils/seeding.py), included in the `gym_v0_12_1_seeding.py` file in the `validation` folder of the [stable-gym](htts://github.com/rickstaa/stable-gym/tree/han2020/tests/manual/validation) package. Although this validation is not critical, as the reset method primarily generates a random state within the observation space without involving system dynamics, it is conducted for thoroughness and completeness of the validation process.

1. **Set a breakpoint in the environment reset method**: Add a breakpoint in the reset method of the environment you are validating, specifically after the `super().reset()` call.

2. **Replace the seeding method**: Swap the seeding method in the `stable-gym` environment with the original OpenAI Gym version. To do this, add the `gym_v0_12_1_seeding.py` file to your `PYTHONPATH` and run the following code in the debug terminal:

   ```python
   from gym_v0_12_1_seeding import np_random
   self.np_random = np_random(seed)[0]
   ```

3. **Set a breakpoint in the original environment reset method**: Insert a breakpoint in the reset method of the original environment for comparison.

4. **Compare the observations**: Evaluate the observations from the original environment against those of the stable-gym environment. If they match, the translation is accurate. Otherwise, further adjustments are necessary.

## Note

- Ensure identical step counts and seed usage in the original and validation scripts for an accurate comparison.
- Adjust the tolerance level in the script as needed for your validation precision requirements.
