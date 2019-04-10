###########################
# 6.00.2x Problem Set 1: Space Cows 

from ps1_partition import get_partitions
import time

#================================
# Part A: Transporting Space Cows
#================================

def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """

    cow_dict = dict()

    f = open(filename, 'r')
    
    for line in f:
        line_data = line.split(',')
        cow_dict[line_data[0]] = int(line_data[1])
    return cow_dict


# Initialize basic data

cows = load_cows('ps1_cow_data.txt')
limit = 10

# Problem 1

def greedy_cow_transport(cows,limit=10):
    
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    
    #assign dictionary to local variable
    temp_cow_dict = cows.copy()
    
    #transport_order needs to be initialized outside loop
    transport_order = []
    
    loop = 0 #initializing counter - DELETE AFTER
    
    #loop as long as dictionary is not empty
    while temp_cow_dict != {}:
        
        #counter - DELETE AFTER
        loop += 1
        
        #initialize cow weight to 0, transport to empty list, weight limit
        cow_weight = 0
        transport = []
        weight_limit = limit
        
        #loop as long as room in transport
        while True:
            
            #initialize cow_weight to 0 for for-loop
            cow_weight = 0
            
            #step down current version of dictionary
            for each_cow in temp_cow_dict:
                
                #compare each cow to max cow weight BUT make sure cow can fit
                if temp_cow_dict[each_cow] > cow_weight and temp_cow_dict[each_cow] <= weight_limit:
                    
                    #assign higher weight to cow_weight
                    cow_weight = temp_cow_dict[each_cow]
                    
                    #assign cow name to heaviest_cow
                    heaviest_cow = each_cow
            
            #if cow weight = 0, means no cows can fit, therefore transport
            if cow_weight == 0:
                transport_order.append(transport)
                break
            
            else:
                
                #add heaviest cow from for-loop into transport
                transport.append(heaviest_cow)
                        
                #lower next weight limit
                weight_limit -= cow_weight
                        
                #remove cow from dictionary and reset cow weight
                del temp_cow_dict[heaviest_cow]

    print(transport_order)


# Problem 2

def brute_force_cow_transport(cows,limit=10):
    
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    
    #create list of cows in descending order
    temp_cow_dict = [(k,v) for v,k in sorted([(v,k) for k,v in cows.items()],reverse=True)]

    #get all partitions of temp_cow_dict
    for partition in get_partitions(temp_cow_dict):
        
        #check if length of current partition is shorter
        if len(partition) > len(temp_cow_dict):
            
            #skips if unnessary to even check
            pass
        
        #runs if length shorter
        else:
        
            #separate each partition into constituent lists
            for each_list in partition:
            
                #set temp_weight equal to 0
                temp_weight = 0
            
                #get weight of each cow that is within partition from temp_cow_dict
                for each_cow in each_list:
                    
                    #get weight of each cow and add to temp_weight
                    temp_weight += each_cow[1]
                    
                #test temp_weight against limit
                if temp_weight > limit:
                    
                    #breaks out of for loop to check next partition
                    break
            
            #checks temp weight again in case we broke out from being over weight
            if temp_weight <= limit:

                #reassigns temp_cow_dict if everything ok until here
                temp_cow_dict = partition

    #prints answer
    print(temp_cow_dict)


# Problem 3
def compare_cow_transport_algorithms():
    
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    
    #time greedy_cow_transport
    start_greedy = time.time()
    greedy_cow_transport(cows, limit)
    end_greedy = time.time()
    print('Greedy Version:',end_greedy - start_greedy)

    print()

    #time brute_force_cow_transport
    start_brute_force = time.time()
    brute_force_cow_transport(cows, limit)
    end_brute_force = time.time()
    print('Brute Force Version:',end_brute_force - start_brute_force)


"""
Here is some test data for you to see the results of your algorithms with. 
Do not submit this along with any of your answers. Uncomment the last two
lines to print the result of your problem.
"""

print('Problem 1')
greedy_cow_transport(cows, limit)

print('\nProblem 2')
brute_force_cow_transport(cows, limit)

print('\nProblem 3')
compare_cow_transport_algorithms()
