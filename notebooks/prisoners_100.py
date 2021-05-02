import random
import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
import math


def get_flat_list(cycles):
    if cycles:
        return [element for cycle in cycles for element in cycle]
    else:
        return []


def get_cycles(choices_org):
    choices = choices_org.copy()
    find_num = 0
    cycles = []
    while not all([i == -1 for i in choices]):
        cycle = []
        slot_num = choices[find_num]
        while (find_num not in get_flat_list(cycles)) and (find_num not in cycle):
            cycle.append(slot_num)
            choices[choices.index(slot_num)] = -1
            slot_num = choices[slot_num]
        find_num += 1
        if cycle:
            cycles.append(cycle)
    return cycles


def simulate_cycle(n):
    choices = list(range(n))
    random.shuffle(choices)
    cycles = get_cycles(choices)
    return cycles, choices


calc_lengths = lambda cycles: np.array([len(cycle) for cycle in cycles])


def simulate_cycles(n, n_trials=1000):
    trials_cycles = [simulate_cycle(n) for trial in range(n_trials)]
    trials = [sequence for cycles, sequence in trials_cycles]
    cycles = [cycles for cycles, sequence in trials_cycles]
    df_results = pd.DataFrame(dict(trial=trials, cycle=cycles))
    df_results["cycle_lengths"] = df_results.cycle.apply(
        lambda trial: [len(cycle) for cycle in trial]
    )
    df_results["num_cycles"] = df_results.cycle.apply(len)
    df_results["avg_cycle_len"] = df_results.cycle_lengths.apply(np.mean)
    return df_results


def calc_naive_prob(n, thresh):
    return sum([1 / i for i in list(range(n - thresh + 1, n + 1))]) ** n


st.title("The 100 prisoners problem")

st.sidebar.markdown("#### Simulation Parameters")
st.sidebar.markdown("")
n = st.sidebar.number_input(
    "Number of prisoners:", min_value=3, max_value=1000, value=10
)
n_trials = st.sidebar.slider(
    "Number of trials to simulate:", min_value=10, max_value=10000, value=1000
)
n_thresh = st.sidebar.slider(
    "Threshold for success", min_value=0, max_value=n, value=math.ceil(n / 2)
)
submit = st.sidebar.button("Simulate!")

naive_prob_replmcnt = (n_thresh / n) ** n
naive_prob_wtt_replmcnt = calc_naive_prob(n, n_thresh)
st.sidebar.markdown(
    f"The naive probability of success is `{naive_prob_replmcnt*100:.3f}%` (with replacement) and `{naive_prob_wtt_replmcnt*100:.3f}%` (without replacement)"
)

if submit:
    df_results = simulate_cycles(n, n_trials)
    df_results["is_success"] = df_results.cycle_lengths.apply(
        lambda trial_lens: (np.array(trial_lens) <= n_thresh).all()
    )

    st.markdown(
        f"### The probability of success using the optimum strategy is `{df_results['is_success'].mean()*100:.2f}%`"
    )

    st.markdown("### Distribution of Number of Cycles")
    st.altair_chart(
        alt.Chart(df_results)
        .mark_bar()
        .encode(alt.X("num_cycles:Q", bin=False), y="count()", color="is_success")
    )

    st.markdown("### Distribution of Average Cycle Length")
    st.altair_chart(
        alt.Chart(df_results)
        .mark_bar()
        .encode(alt.X("avg_cycle_len:Q", bin=False), y="count()", color="is_success")
    )

    # st.altair_chart(alt.Chart(df_results).mark_circle(size=60).encode(
    #     x='num_cycles',
    #     y='avg_cycle_len',
    #     color='is_success',
    #     tooltip=['num_cycles', 'avg_cycle_len', 'is_success',]
    # ).transform_calculate(
    #     # Generate Gaussian jitter with a Box-Muller transform
    #     jitter='sqrt(-2*log(random()))*cos(2*PI*random())'
    # ).interactive())

    st.markdown("### Detailed Simulation Info")
    st.write(df_results)

    st.markdown("### Solution Explained")
    st.video("https://www.youtube.com/embed/vIdStMTgNl0", start_time=103)
else:
    st.markdown("Click on simulate to run a new simulation. ")
    st.markdown(
        "To view the Ted-ed version of the puzzle, please click this [link](https://www.youtube.com/vIdStMTgNl0). For the original puzzle, refer to [this wikipedia page](https://en.wikipedia.org/wiki/100_prisoners_problem)"
    )
