from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import pandas as pd

def plot(all_rewards, smoothed_rewards):
    plt.figure(2, figsize=(12, 6))
    plt.clf()
    plt.title("Training")
    plt.xlabel("Epsiode")
    plt.ylabel("Total Reward")
    plt.plot(all_rewards, '--', alpha=0.5)
    plt.plot(smoothed_rewards)
    plt.legend(["Rewards", "Rewards (Smoothed)"])
    plt.pause(0.0001)
    display.clear_output(wait=True)

def qlearning_nn(agent, policy_network, target_network, num_episodes, window_size=10, strategy=None, memory=None):

    if agent.is_learning() and memory is None:
        print("Agent is learning, function requires memory")
    if agent.is_learning() and strategy is None:
        print("Agent is learning, function requires strategy")
    
    discount_factor = 0.95
    env = agent.env
    
    all_rewards = []
    for episode in range(num_episodes):
        done = False
        rewards = []
        exploits = 0
        
        state = agent.preprocess_state(env.reset())
        
        while not done:

            action, exploit = agent.get_action(strategy, policy_network, state)
            if exploit:
                exploits+=1
                

            new_state, reward, done, i = env.step(action=action)
            new_state = agent.preprocess_state(new_state)
            rewards.append(reward)
            
            if agent.is_learning():
                memory.push(state, action, new_state, reward)
        
            if done:
                if agent.is_learning():
                    if memory.can_sample():
                        # Fetch a batch from the replay buffer and extract as numpy arrays 
                        train_rewards, train_states, train_new_state, train_actions, batch_size = memory.extract_samples()

                        q_values_for_current_state = policy_network.predict(train_states.reshape(batch_size,agent.get_num_states())) # predict current values for the given states
                        q_values_for_new_state     = target_network.predict(train_new_state.reshape(batch_size,agent.get_num_states()))                    
                        q_values_for_current_state_tmp = train_rewards + discount_factor * np.amax(q_values_for_new_state,axis=1)                
                        q_values_for_current_state[ (np.arange(batch_size), train_actions.reshape(batch_size,).astype(int))] = q_values_for_current_state_tmp                                                                              
                        policy_network.update(train_states.reshape(batch_size,agent.get_num_states()), q_values_for_current_state) # Update the function approximator 
                
                if episode % 20 == 0:
                    target_network.model.set_weights(policy_network.model.get_weights())
                total_reward = np.sum(rewards)
                all_rewards.append(total_reward)
                
                ## taking moving average of rewards to smooth
                smoothed_rewards = pd.Series(all_rewards).rolling(window_size, min_periods=window_size).mean()
                this_smoothed_reward = smoothed_rewards.values[-1]
                
                        
                if num_episodes == 1:
                    print("Evaluation reward " + str(total_reward))
                else:
                    print("Episode = " + str(episode) + ". Num Exploits = " + str(exploits) + ". Total Reward = " + str(total_reward)
                          + ". Moving Average Reward = " + str(this_smoothed_reward))
                    plot(all_rewards, smoothed_rewards)


                
            state = new_state
            if strategy:
                strategy.episode_complete()
    
    return all_rewards