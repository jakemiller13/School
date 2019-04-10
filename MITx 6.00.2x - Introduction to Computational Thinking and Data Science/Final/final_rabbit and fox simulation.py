import random
import pylab

# Global Variables
#MAXRABBITPOP = 1000
#CURRENTRABBITPOP = 500
#CURRENTFOXPOP = 30

MAXRABBITPOP = 1000
CURRENTRABBITPOP = 50
CURRENTFOXPOP = 300

def rabbitGrowth():
    """ 
    rabbitGrowth is called once at the beginning of each time step.

    It makes use of the global variables: CURRENTRABBITPOP and MAXRABBITPOP.

    The global variable CURRENTRABBITPOP is modified by this procedure.

    For each rabbit, based on the probabilities in the problem set write-up, 
      a new rabbit may be born.
    Nothing is returned.
    """
    # you need this line for modifying global variables
    global CURRENTRABBITPOP
    
    for each in range(CURRENTRABBITPOP):
        if CURRENTRABBITPOP < MAXRABBITPOP and random.random() < (1.0 - CURRENTRABBITPOP/MAXRABBITPOP):
            CURRENTRABBITPOP += 1


def foxGrowth():
    """ 
    foxGrowth is called once at the end of each time step.

    It makes use of the global variables: CURRENTFOXPOP and CURRENTRABBITPOP,
        and both may be modified by this procedure.

    Each fox, based on the probabilities in the problem statement, may eat 
      one rabbit (but only if there are more than 10 rabbits).

    If it eats a rabbit, then with a 1/3 prob it gives birth to a new fox.

    If it does not eat a rabbit, then with a 1/10 prob it dies.

    Nothing is returned.
    """
    # you need these lines for modifying global variables
    global CURRENTRABBITPOP
    global CURRENTFOXPOP
    
    new_foxes = 0
    
    for each in range(CURRENTFOXPOP):
        if CURRENTRABBITPOP > 10:
            if random.random() < (CURRENTRABBITPOP/MAXRABBITPOP):
                CURRENTRABBITPOP -= 1
                if random.random() < 1/3:
                    new_foxes += 1
            else:
                if random.random() < 0.9:
                    new_foxes -= 1
                
    CURRENTFOXPOP += new_foxes

            
def runSimulation(numSteps):
    """
    Runs the simulation for `numSteps` time steps.

    Returns a tuple of two lists: (rabbit_populations, fox_populations)
      where rabbit_populations is a record of the rabbit population at the 
      END of each time step, and fox_populations is a record of the fox population
      at the END of each time step.

    Both lists should be `numSteps` items long.
    """
    
    rabbit_populations = []
    fox_populations = []
    
    for step in range(numSteps):
        if CURRENTRABBITPOP <= MAXRABBITPOP:
            rabbitGrowth()
        if CURRENTFOXPOP > 10:
            foxGrowth()
        rabbit_populations.append(CURRENTRABBITPOP)
        fox_populations.append(CURRENTFOXPOP)
    
    pylab.figure()
    pylab.plot(rabbit_populations,label = 'Current Rabbit Population')
    pylab.plot(fox_populations, label = 'Current Fox Population')
    pylab.legend(loc = 'best')
    
    rabbit_coeff = pylab.polyfit(range(len(rabbit_populations)),rabbit_populations,2)
    fox_coeff = pylab.polyfit(range(len(fox_populations)),fox_populations,2)
    print('rabbit coefficients = ',rabbit_coeff)
    print('fox coefficients = ',fox_coeff)
    
    pylab.figure()
    pylab.plot(pylab.polyval(rabbit_coeff, range(len(rabbit_populations))))
    
    pylab.figure()
    pylab.plot(pylab.polyval(fox_coeff, range(len(fox_populations))))
    
    return (rabbit_populations, fox_populations)

runSimulation(200)
#print(runSimulation(200))