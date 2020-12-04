def epsilon_with(decay, episodes):
    print("Epsilon with decay " + str(decay) + " and " + str(episodes) + " episodes")
    check_epsilon_after(decay, int(episodes/4))    
    check_epsilon_after(decay, int(episodes/2))    
    check_epsilon_after(decay, int((episodes*3)/4))    
    check_epsilon_after(decay, int(episodes))
    print("------")

 

def check_epsilon_after(decay, episodes):
    #illustrates the effect that of the power low decay
    value = (decay**52)**episodes
    print("After " + str(episodes) + " episodes, epsilon will be " + str(value))

 

epsilon_with(0.999985, 2000)