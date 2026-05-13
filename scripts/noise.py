import random
import pandas as pd


class NoiseToProxyVoters:
    """Applies random noise to proxy voter responses to simulate response uncertainty."""

    def __init__(
        self,
        noise=None,
        random_state=42,
        first_iteration_noise_multiplier=3.0,
    ):
        self.noise = noise
        self.base_seed = random_state
        self.first_iteration_noise_multiplier = first_iteration_noise_multiplier

    def iterative_noise(self, proxy_voters_mapped, iterations=5, alpha=None):
        """Apply noise iteratively, with each iteration building on the previous output.

        Args:
            proxy_voters_mapped: DataFrame with a 'Svar_mapped' column of numeric responses
            iterations: number of noise iterations to run
            alpha: scale parameter matching the response mapping

        Returns:
            Tuple of (concatenated noisy DataFrame with 'noise_id' column, total switch count).
        """
        noisy_dataframes = []
        total_switch_count = 0
        proxy_voters = proxy_voters_mapped.copy()
        for i in range(iterations):
            # Apply higher noise in the first iteration
            current_noise = (
                self.noise * self.first_iteration_noise_multiplier
                if i == 0
                else self.noise
            )
            proxy_voters, switch_count = self.add_noise(
                proxy_voters,
                current_noise,
                None,
                alpha,
                random_state=self.base_seed + i,
            )
            proxy_voters["noise_id"] = i + 1
            total_switch_count += switch_count
            noisy_dataframes.append(proxy_voters)

        proxy_voters_noise = pd.concat(noisy_dataframes, ignore_index=True)

        return proxy_voters_noise, total_switch_count

    def iterative_noise_from_original(
        self,
        proxy_voters_mapped,
        iterations=1,
        noise_rounds_per_question=None,
        alpha=None,
    ):
        """Apply noise independently from the original data each iteration (non-compounding).

        Args:
            proxy_voters_mapped: DataFrame with a 'Svar_mapped' column of numeric responses
            iterations: number of independent noisy copies to produce
            noise_rounds_per_question: if set, applies multiple adjacent-step perturbations per question
            alpha: scale parameter matching the response mapping

        Returns:
            Tuple of (concatenated noisy DataFrame with 'noise_id' column, total switch count).
        """
        noisy_dataframes = []
        total_switch_count = 0
        original = proxy_voters_mapped.copy()
        for i in range(iterations):
            random_state = self.base_seed + i
            noisy, switch_count = self.add_noise(
                original, self.noise, noise_rounds_per_question, alpha, random_state
            )
            noisy["noise_id"] = i + 1
            total_switch_count += switch_count
            noisy_dataframes.append(noisy)

        proxy_voters_noise = pd.concat(noisy_dataframes, ignore_index=True)
        return proxy_voters_noise, total_switch_count

    def add_noise(
        self,
        proxy_voters_mapped,
        noise_level,
        noise_rounds_per_question,
        alpha,
        random_state=None,
    ):
        """Perturb each response with probability noise_level by shifting to an adjacent value.

        Args:
            proxy_voters_mapped: DataFrame with a 'Svar_mapped' column of numeric responses
            noise_level: probability in [0, 1] that any given response is perturbed
            noise_rounds_per_question: if set, applies this many adjacent-step shifts per perturbed response;
                if None, uses the original single-step perturbation logic
            alpha: scale parameter determining the extreme response values (±(1 + alpha))
            random_state: seed for the random generator; falls back to self.base_seed if None

        Returns:
            Tuple of (noisy copy of proxy_voters_mapped, number of responses that changed).
        """
        random_state = self.base_seed if random_state is None else random_state
        switch_count = 0
        proxy_voters_noise = proxy_voters_mapped.copy()
        proxy_voters_noise["Svar_mapped"] = proxy_voters_noise["Svar_mapped"].astype(
            float
        )
        random_generator = random.Random(random_state)

        if noise_rounds_per_question is not None:
            new_values = []
            for original_value in proxy_voters_mapped["Svar_mapped"]:
                new_value = original_value
                if random_generator.random() < noise_level:
                    intermediate_value = original_value
                    for i in range(noise_rounds_per_question):
                        if intermediate_value == 1 + alpha:
                            intermediate_value = (
                                1 if random_generator.random() < 0.5 else 1 + alpha
                            )
                        elif intermediate_value == 1:
                            intermediate_value = (
                                1 + alpha if random_generator.random() < 0.5 else 0
                            )
                        elif intermediate_value == 0:
                            intermediate_value = (
                                1 if random_generator.random() < 0.5 else -1
                            )
                        elif intermediate_value == -1:
                            intermediate_value = (
                                -1 - alpha if random_generator.random() < 0.5 else 0
                            )
                        elif intermediate_value == -1 - alpha:
                            intermediate_value = (
                                -1 if random_generator.random() < 0.5 else -1 - alpha
                            )

                        if (
                            i == (noise_rounds_per_question - 1)
                            and intermediate_value == 0
                        ):
                            intermediate_value = (
                                1 if random_generator.random() < 0.5 else -1
                            )

                    new_value = intermediate_value

                if new_value != original_value:
                    switch_count += 1
                new_values.append(new_value)
            proxy_voters_noise["Svar_mapped"] = new_values

        return proxy_voters_noise, switch_count
