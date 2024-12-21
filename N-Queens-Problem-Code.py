import random
import time 
import pandas as pd
import matplotlib.pyplot as plt


# Generate k random initial states 
def generateInitialStates(n, k):
    return [random.sample(range(1, n + 1), n) for x in range(k)]

# Calculate the heuristic value of a state (number of attacking pairs)
def calculateHeuristic(state):
    n = len(state)
    conflicts = 0
    # Check all pairs of queens for conflicts
    for i in range(n):
        for j in range(i + 1, n):
            # Conflict if queens are in the same row or on the same diagonal
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                conflicts += 1
    return conflicts 

# Generate all possible successor states by moving queens in each column
def generateSuccessors(state):
    n = len(state)
    successors = []
    for col in range(n):  # Iterate over all columns
        for row in range(1, n + 1):  # Try moving the queen to all rows in this column
            if row != state[col]:  # Avoid generating the current state
                newState = state[:]  # Copy the current state
                newState[col] = row  # Move the queen to a new row
                successors.append(newState)  # Add the new state to the successors
    return successors

# Implement the Local Beam Search algorithm
def localBeamSearch(n, k):
    startTime = time.time()  # Record the start time
    states = generateInitialStates(n, k)  # Generate k random initial states
    
    # Track the best state and its heuristic from the previous iteration
    prev_best_state = None
    prev_best_heuristic = float('inf')  # Start with a very large heuristic value, could possibly set to -1 as well.
    
    while True:
        # Evaluate heuristics for all current states
        statesWithHeuristics = [(state, calculateHeuristic(state)) for state in states]

        # Check if any state is a solution (heuristic == 0)
        for state, heuristic in statesWithHeuristics:
            if heuristic == 0:  # Solution found
                endTime = time.time()  # Record the end time
                return {
                    "solution": state,  # Return the solution state
                    "time": endTime - startTime,  # Total time taken
                    "heuristic": heuristic,  # Heuristic value of the solution (should be 0)
                    "finalState": state  # Final state is the solution
                }

        # Generate all successors for the current states
        successors = []
        for state, _ in statesWithHeuristics:
            successors.extend(generateSuccessors(state))

        # Evaluate successors and sort by heuristic
        successorsWithHeuristics = [(s, calculateHeuristic(s)) for s in successors]
        successorsWithHeuristics.sort(key=lambda x: x[1])  # Sort by heuristic 

        # Store the best state and heuristic from the current iteration
        current_best_state = successorsWithHeuristics[0][0]  
        current_best_heuristic = successorsWithHeuristics[0][1]  

        # Update the states with the best k successors
        states = [s[0] for s in successorsWithHeuristics[:k]]  

        # Check if no progress is being made 
        if current_best_heuristic >= prev_best_heuristic:
            endTime = time.time()
            return {
                "solution": None,  # No solution found
                "time": endTime - startTime,  # Total time taken
                "heuristic": prev_best_heuristic,  # Best heuristic from previous iteration
                "finalState": prev_best_state  # Best state from previous iteration
            }

        # Update the previous best state and heuristic for the next iteration
        prev_best_state = current_best_state
        prev_best_heuristic = current_best_heuristic


# Plot runtime vs. k for each n
def plotRuntime(df):
    for n in [8, 12, 16, 20]:
        # Calculate average runtime for each k and n
        runtimes = df[df['n'] == n].groupby('k')['time'].mean()
        plt.plot(runtimes.index, runtimes.values, label=f"n={n}")

    plt.xlabel("k (number of states)")
    plt.ylabel("Average Runtime (seconds)")
    plt.legend()
    plt.title("Average Runtime vs. k")
    plt.show()


# Run experiments with different values of n and k
def runExperiments():
    results = []
    for n in [8, 12, 16, 20]:  # Test for n
        for k in [1, 3, 5, 7]:  # Test with k
            for trial in range(3):  # Run trials for each combination of n and k
                result = localBeamSearch(n, k)
                results.append({
                    "n": n,  # Number of queens
                    "k": k,  # Number of states tracked in the beam
                    "trial": trial + 1,  # Trial number
                    "solutionFound": result["solution"] is not None,  # True if a solution was found
                    "heuristic": result["heuristic"],  # Best heuristic value
                    "time": result["time"],  # Time taken
                    "finalState": result.get("finalState", [])  # Best state found
                })
    return results

# Analyze the results 
def analyzeResults(results):
    df = pd.DataFrame(results)  # Convert results to a DataFrame
    print(df.head(100))  # Preview the first 100 rows of the data
    
    # Plot success rate vs. k for each n
    for n in [8, 12, 16, 20]:
        successRates = df[df['n'] == n].groupby('k')['solutionFound'].mean()
        plt.plot(successRates.index, successRates.values, label=f"n={n}")
    
    plt.xlabel("k (number of states)")
    plt.ylabel("Success Rate")
    plt.legend()
    plt.title("Success Rate vs. k")
    plt.show()


# Run the experiments and analyze results
results = runExperiments()  # Run experiments
analyzeResults(results)  # Analyze success rates
df = pd.DataFrame(results)  # Create a DataFrame from results
plotRuntime(df)  # Plot runtime trends
