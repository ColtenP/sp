import json
import math
import statistics
import sys
from multiprocessing import Pool, cpu_count
from textwrap import dedent

import numpy as np
import tqdm
from scipy.stats import lognorm
from scipy.stats import nbinom
from scipy.stats import norm

# Parameters for Negative Binomial (NB) distribution
rNB = 26  # r parameter
bNB = 0.568  # Beta parameter

# Parameters for CLGP distribution
theta = 1.1318995
sigma = 0.1775174
alpha = 1.5726146
lamda = 0.3848036

# Parameters for Premium Pricing
d = 1  # deductible
rho = 0.3  # loading factor
c_q = theta  # should be higher than deductible

# Parameters for Surplus Calculation
u_0 = 5  # initial surplus
v_0 = 5  # initial storage
n = 100  # number of periods per one simulation in monte carlo

# Parameters for Ruin
dly = 2

mu = (np.log(theta)) - ((sigma ** 2) * (((alpha * theta) - lamda) / (lamda + theta)))

w_num_div = (
    (math.sqrt(2 * np.pi)) * alpha * theta * sigma *
    (norm.cdf((np.log(theta) - mu) / sigma)) *
    (math.exp((((np.log(theta) - mu) / sigma) ** 2) / 2))
)
w = w_num_div / (w_num_div + lamda + theta)


def LN_pdf(x, sigma, mu):  # pdf of Log normal distribution
    return (1 / ((math.sqrt(2 * np.pi)) * x * sigma)) * \
          (math.exp(-(((np.log(x) - mu) / sigma) ** 2) / 2))


def LN_cdf(x, sigma, mu):  # pdf of Log normal distribution
    return norm.cdf((np.log(x) - mu) / sigma)


def LN_inv(x, sigma, mu):
    LN_mean = math.exp(mu + ((sigma ** 2) / 2))
    LN_sd = LN_mean * (math.sqrt((math.exp(sigma ** 2)) - 1))
    return lognorm.ppf(x, LN_sd, loc=0, scale=np.exp(mu))


def GPD_pdf(x, theta, alpha, lamda):
    return (alpha * ((lamda + theta) ** alpha)) / ((lamda + x) ** (alpha + 1))


def GPD_cdf(x, theta, alpha, lamda):
    return -(((lamda + theta) / (lamda + x)) ** alpha)


def GPD_inv(x, theta, alpha, lamda):
    beta = (lamda + theta) / alpha
    return (alpha * beta) * ((1 / (1 - x)) - 1) + theta


def CLGP_cdf(x, theta, alpha, lamda, sigma, mu, w):
    if theta > x > 0:
        return w * ((LN_cdf(x, sigma, mu)) / (LN_cdf(theta, sigma, mu)))
    else:
        return 1 - ((1 - w) * (((lamda + theta) / (lamda + x)) ** alpha))


def generate_output_document(run_result):

    return dedent(f'''
        Line of Business: FIRE I
        Premium Principles: Tailed Normal Standard Deviation
        
        {' Parameters '.center(60, '-')}
        
        rNB = {rNB}
        bNB = {bNB}
        thetha = {theta}
        sigma = {sigma}
        alpha = {alpha}
        lamda = {lamda}
        d = {d}
        rho = {rho}
        u_0 = {u_0}
        v_0 = {v_0}
        n = {n}
        N = {run_result['run_count']}
        dly = {dly}
        
        {' Results '.center(60, '-')}
        
        Number of Runs: {run_result['run_count']}
        
        Number of Classical Ruins: {run_result['classical_ruins']}
        Number of Parisian Ruins: {run_result['parisian_ruins']}
        
        Classical Ruin Probability: {run_result['classical_ruin_probability']}
        Parisian Ruin Probability: {run_result['parisian_ruin_probability']}
        
        Expected Storage: {run_result['expected_storage']}
        Markov Bound: {run_result['markov_bound']}
        Alternative Markov Bound: {run_result['alternative_markov_bound']}
    ''')[1:-1]


def calculate_loss_for_insurer_and_reinsurer(run_count, iteration, o):
    np.random.seed(run_count * iteration * o)
    k = np.random.uniform(low=0, high=1, size=1)[0]  # array

    if k <= w:
        x1 = (k / w) * (LN_cdf(theta, sigma, mu))
        x = LN_inv(x1, sigma, mu)
        c = max(x - c_q, 0)
    else:
        x2 = (k - w) / (1 - w)
        x = GPD_inv(x2, theta, alpha, lamda)
        c = max(x - c_q, 0)

    # loss for insurer, loss for reinsurer
    return x, c


def calculate_pos_and_neg_count_and_loss_reinsurer_sum(loss_reinsurer):
    positive, negative, loss_reinsurer_sum = 0, 0, []

    for loss in loss_reinsurer:
        if loss <= 0:
            negative += 1
            continue

        positive += 1
        loss_reinsurer_sum.append(loss)

    return positive, negative, loss_reinsurer_sum


def calculate_HZtsd(Z, positive_count, loss_reinsurance_sum):
    if positive_count == 0:
        return 0

    C_bar = Z / positive_count

    if positive_count > 1:
        var = statistics.variance(loss_reinsurance_sum)
    else:
        var = 0

    # Tailed Standard Deviation Premium Principle
    if C_bar == 0:
        TSDez = 0

    # Case (I): 0<C_bar<theta
    elif 0 < C_bar < theta:
        erf1 = math.erf((mu-np.log(c_q)) / ((2 ** (1 / 2)) * sigma))
        erf2 = math.erf((mu+(sigma**2)-(np.log(c_q))) / ((2 ** (1 / 2)) * sigma))
        TSDez = (
            (rNB * bNB) * w * (1 / (norm.cdf((np.log(theta) - mu) / sigma))) * (((1 / 2) * d) -
            ((1 / 2) * (math.exp(mu + ((sigma ** 2) / 2)))) - ((1 / 2) * d * erf1) + ((1 / 2) *
            (math.exp(mu + ((sigma ** 2) / 2))) * erf2))
        )

    # Case (II): theta<x<infty
    else:
        TSDez = (
            (rNB*bNB)*(1-w)*((alpha*((theta+lamda)**alpha)*((lamda+c_q)**(-alpha))
                              * ((-alpha*d)+d+lamda+(alpha*c_q)))/(alpha*(alpha-1)))
        )

    return TSDez + (rho * var)


def write_run_result_to_disk(run_result):
    with open(f'./results/{run_result["run_count"]}.json', 'wt') as f:
        f.write(json.dumps(run_result))

    with open(f'./results/{run_result["run_count"]}.txt', 'wt') as f:
        f.write(generate_output_document(run_result))


def run_monte_carlo_round(run_count):
    # Setup Initial Storage Variables
    surplus = [u_0]  # array for previous surplus
    storage = [v_0]
    surplus_premium = [0]  # array for aggregate reinsurance's premiums
    surplus_losses = []  # array for aggregate reinsurance's losses

    for j in range(1, n):
        # Generate random numbers from NB:
        y = nbinom.rvs(rNB, bNB, size=1, random_state=run_count * j)[0]  # Number of claims for INSURER

        # array for claims severity or claims losses of the INSURER
        # array for claims severity or claims losses of the REINSURER
        results_ins_rei = [calculate_loss_for_insurer_and_reinsurer(run_count, j, o) for o in range(0, y)]
        loss_reinsurance = [i for _, i in results_ins_rei]

        Z = sum(loss_reinsurance)  # Aggregate reinsurance claims loss

        [pos_count, _, loss_rei_summ] = calculate_pos_and_neg_count_and_loss_reinsurer_sum(loss_reinsurance)

        # If there are no reinsurance losses
        HZtsd = calculate_HZtsd(Z, pos_count, loss_rei_summ)

        surplus_losses.append(Z)
        surplus_premium.append(HZtsd)

        U = u_0 + (sum(surplus_premium)) - (sum(surplus_losses))
        V = v_0 + (sum(surplus_losses)) - (sum(surplus_premium))

        surplus.append(U)
        storage.append(max(V, 0))

    # If a classical ruin occurred in a single n-period, then we produce a 1, if not, then produce a 0
    c_ruin = 1 if any([num < 0 for num in surplus]) else 0

    # If a parisian ruin occurred in a single n-period, then we produce a 1, if not, then produce a 0
    p_ruin = 1 if any([surplus[idx] < 0 and all(storage[idx:(idx + dly)]) > 0 for idx in range(0, len(storage))]) else 0

    # c_ruin, p_ruin, A
    return c_ruin, p_ruin, sum(storage) / n


def main(stops):
    total_runs = max(stops)

    # Create a Multiprocessing Pool, this will allow us to run multiple simulations at the same time, rather than
    # having to complete one simulation before starting the next, once the simulations are complete we then can
    # take each simulation's result then calculate the result
    with Pool(cpu_count()) as pool:
        c_ruin_running_sum = 0
        p_ruin_running_sum = 0
        running_v_sum = 0
        running_sum = 0
        running_alternative_markov = sys.maxsize

        # For every result, increase the running values to calculate the run_result at each run_count
        for idx, sim in enumerate(tqdm.tqdm(pool.imap(run_monte_carlo_round, range(total_runs)), total=total_runs)):
            run_count = idx + 1
            c_ruin, p_ruin, A = sim

            c_ruin_running_sum += c_ruin
            p_ruin_running_sum += p_ruin
            running_v_sum += A
            running_sum += A
            running_alternative_markov = min(running_alternative_markov, running_sum / u_0)

            c_ruin_probability = c_ruin_running_sum / run_count
            p_ruin_probability = p_ruin_running_sum / run_count
            EV = running_v_sum / run_count
            markov_bound = EV / u_0
            # Chernoff = min(Chernoff)

            if run_count in stops:
                run_result = {
                    'run_count': run_count,
                    'classical_ruins': c_ruin_running_sum,
                    'parisian_ruins': p_ruin_running_sum,
                    'classical_ruin_probability': c_ruin_probability,
                    'parisian_ruin_probability': p_ruin_probability,
                    'expected_storage': EV,
                    'markov_bound': markov_bound,
                    'alternative_markov_bound': running_alternative_markov
                }

                write_run_result_to_disk(run_result)


if __name__ == '__main__':
    main([100, 1_000, 10_000, 100_000, 1_000_000])
