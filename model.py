import numpy as np
import pandas as pd
import numpy.linalg as linalg
import matplotlib.pyplot as plt

from data_preparation import get_rates, get_decomp


instruments=['usd','rub','fx']
def stoch_wrapper(decomp):
    def make_stoch(num):
        sigma=[0.03, 0.0093, 0.11]
        stoch_generator = np.dot(np.random.normal(size=(num,3)),decomp)*sigma
        return stoch_generator
    return make_stoch


stoch_generator = stoch_wrapper(get_decomp())

def simulate_hull_white(
    curve_rub, 
    curve_usd, 
    curve_fx,
    decomp,#decomposed matrix of covariation
    init, #init values for rub rate, usd rate, fx rate
    rub_alpha=1,
    usd_alpha=1,
    sim_number = 10,
    ):
    sigma=[0.03, 0.0093, 0.11]

    k_fx=0.015
    dt=14/365
    

    timesteps = 27
    results = np.zeros(shape=(timesteps, 3, sim_number))


    for sim_ix in range(sim_number):
        results[0,:,sim_ix] = init

        stochs = stoch_generator(timesteps)
        for i, (rate_rub, rate_usd, rate_fx, stoch_tuple) in enumerate(zip(curve_rub,curve_usd,curve_fx, stochs)):
            results[i+1,0,sim_ix] = results[i,0,sim_ix] + (rate_rub - rub_alpha*results[i,0,sim_ix])*dt+stoch_tuple[0]
            results[i+1,1,sim_ix] = results[i,1,sim_ix] + (rate_usd - usd_alpha*results[i,1,sim_ix])*dt+stoch_tuple[1]
            results[i+1,2,sim_ix] = results[i,2,sim_ix] + k_fx*(rate_fx - np.log(results[i,2,sim_ix]))*dt+stoch_tuple[2]

    return results


def main():

    curve_usd, curve_rub, curve_fx, init = get_rates()


    decomp=get_decomp()
    results = simulate_hull_white(
        curve_rub=curve_rub, 
        curve_usd=curve_usd, 
        curve_fx=curve_fx,
        decomp=decomp,
        init=init,
        rub_alpha=1,
        usd_alpha=1,
        sim_number=10
        )
    # results - np array of shape(no of timesteps, no of instruments, no of simulations)
    plot_results(results)
    return results

def plot_results(result):
    fig,ax = plt.subplots(1,3,figsize=(25,5))
    titles=['Процентная ставка в рублях', "Процентная ставка в доларах", "Обменный курс"]
    for i in range(3):
        ax[i].plot(result[:,i,:], alpha=0.1)
        pd.DataFrame(result[:,i,:].T).quantile(0.05).plot(ax=ax[i])
        ax[i].set_title(titles[i]+ ' (VaR 5%)')
    plt.show()


sigma=[0.03, 0.0093, 0.11]

k_fx=0.015
dt=14/365

sim_number = 10
timesteps = 27
curve_usd, curve_rub, curve_fx, init = get_rates()

dict_of_rates={
    'usd':curve_usd,
    'rub':curve_rub,
    'fx':curve_fx,
}


rub_alpha=1
usd_alpha=1

dict_of_getters={
    'rub':lambda prev_state, rate_rub, stoch:  (rate_rub - rub_alpha*prev_state)*dt+stoch[0],
    'usd':lambda prev_state, rate_usd, stoch:  (rate_usd - usd_alpha*prev_state)*dt+stoch[1],
    'fx':lambda prev_state, rate_fx,  stoch:   k_fx*(rate_fx - np.log(prev_state))*dt+stoch[2],

}


#id tuple:  rate_type, timestep, value


# def calc_1_step(rate_type, timestep, value, stoch):
#     if rate_type=='usd':
#         getter = get_new_usd

#     return getter(value, dict_of_rates[rate][timestep], stoch)



"""
mini_step = 2 weeks
large_step - parameter (for now = 5 mini steps)
num_steps - calculated parameter (how many large steps)
simulations_number - parameter (for now = 1000)

1. initialize rates.
2. make sim_num simulations of large_step steps; save results
3. bootstrap sim_num points from previous step, make simulations
4. terminate when num_steps is reached

"""


make_stoch = stoch_wrapper(get_decomp())


def make_large_step(
    ministeps, 
    initial_values, 
    initial_timestep):
    history=[]
    stoch = make_stoch(ministeps)
    for instr_ix in range(3):
        instrument_kind = instruments[instr_ix]
        instrument_history=[(instrument_kind, initial_timestep, initial_values[instr_ix])]
        for i in range(ministeps):

            val_increment = dict_of_getters[instrument_kind](
                instrument_history[-1][-1], 
                dict_of_rates[instrument_kind][initial_timestep+i], 
                stoch[i])
            instrument_history.append(
                (instrument_kind,initial_timestep+i+1, instrument_history[-1][-1]+val_increment)
            )
        history.extend(instrument_history)

    return history


def dispatcher(sims=1000):
    curve_usd, curve_rub, curve_fx, init = get_rates()

    dict_of_rates={
        'usd':curve_usd,
        'rub':curve_rub,
        'fx':curve_fx,
    }
    usd_alpha=1
    rub_alpha=1


    cp_count=5
    ts_per_cp=5
    
    history=[[('usd', 0, init[0],'null'),('rub', 0, init[1],'null'),('fx', 0, init[2],'null'),]]
    for large_step in range(cp_count):
        local_history=[]
        # for instrument_kind in ['usd','rub','fx']:

        """
        we urgently need to introduce simulation index, by which we will bootstrap start values
        we also need to add ix appending to make_large_step
        
        """
        start_values = np.random.choice([x[2] for x in history[-1] if x[0]==instrument_kind], size=sims
            )
        for init_value in start_values:
            local_history.extend(
                make_large_step(
                    instrument_kind,
                    ts_per_cp, 
                    init_value, 
                    large_step*ts_per_cp))
        history.append(local_history)
    return history
"""
todo:
dispatcher for large steps
    must initialize, collect results on each checkpoint, bootstrap last results for new large step
"""
