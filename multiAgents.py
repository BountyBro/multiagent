# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Handle immediate win/loss states
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')

        # Evaluate food proximity
        foodList = newFood.asList()
        if foodList:
            nearestFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
            foodScore = 10.0 / nearestFoodDist
        else:
            foodScore = 0

        # Evaluate ghost threats
        ghostScore = 0
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            
            if newScaredTimes[i] > 0:
                # Scared ghost - treat as opportunity
                ghostScore += 200.0 / (ghostDist + 1)
            else:
                # Normal ghost - penalize proximity heavily
                if ghostDist < 2:
                    return -500
                ghostScore -= 10.0 / (ghostDist + 1)

        # Combine all factors
        return successorGameState.getScore() + foodScore + ghostScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        # Get all legal actions for Pacman (agent index 0)
        legalActions = gameState.getLegalActions(0)
        numAgents = gameState.getNumAgents()
        
        # Find the best action for Pacman
        bestAction = None
        bestScore = float('-inf')
        
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = self.minimaxValue(successor, 0, 1, numAgents)
            if score > bestScore:
                bestScore = score
                bestAction = action
        
        return bestAction
    
    def minimaxValue(self, gameState, depth, agentIndex, numAgents):
        """Recursively compute the minimax value for a given state."""
        # Terminal states
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # Check if we've reached the maximum depth
        if depth == self.depth:
            return self.evaluationFunction(gameState)
        
        # Get legal actions for current agent
        legalActions = gameState.getLegalActions(agentIndex)
        
        # Handle case with no legal actions
        if not legalActions:
            return self.evaluationFunction(gameState)
        
        # Determine next agent and depth
        nextAgent = (agentIndex + 1) % numAgents
        newDepth = depth + 1 if nextAgent == 0 else depth
        
        if agentIndex == 0:
            # Pacman's turn - maximize
            bestScore = float('-inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = self.minimaxValue(successor, newDepth, nextAgent, numAgents)
                bestScore = max(bestScore, score)
            return bestScore
        else:
            # Ghost's turn - minimize
            bestScore = float('inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = self.minimaxValue(successor, newDepth, nextAgent, numAgents)
                bestScore = min(bestScore, score)
            return bestScore
        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        # Get all legal actions for Pacman (agent index 0)
        legalActions = gameState.getLegalActions(0)
        numAgents = gameState.getNumAgents()
        
        # Find the best action for Pacman with alpha-beta pruning
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = self.alphaBetaValue(successor, 0, 1, numAgents, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, score)
        
        return bestAction
    
    def alphaBetaValue(self, gameState, depth, agentIndex, numAgents, alpha, beta):
        """Recursively compute the minimax value with alpha-beta pruning."""
        # Terminal states
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # Check if we've reached the maximum depth
        if depth == self.depth:
            return self.evaluationFunction(gameState)
        
        # Get legal actions for current agent
        legalActions = gameState.getLegalActions(agentIndex)
        
        # Handle case with no legal actions
        if not legalActions:
            return self.evaluationFunction(gameState)
        
        # Determine next agent and depth
        nextAgent = (agentIndex + 1) % numAgents
        newDepth = depth + 1 if nextAgent == 0 else depth
        
        if agentIndex == 0:
            # Pacman's turn - maximize with alpha-beta pruning
            bestScore = float('-inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = self.alphaBetaValue(successor, newDepth, nextAgent, numAgents, alpha, beta)
                bestScore = max(bestScore, score)
                if bestScore > beta:
                    return bestScore
                alpha = max(alpha, bestScore)
            return bestScore
        else:
            # Ghost's turn - minimize with alpha-beta pruning
            bestScore = float('inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = self.alphaBetaValue(successor, newDepth, nextAgent, numAgents, alpha, beta)
                bestScore = min(bestScore, score)
                if bestScore < alpha:
                    return bestScore
                beta = min(beta, bestScore)
            return bestScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        # Get all legal actions for Pacman (agent index 0)
        legalActions = gameState.getLegalActions(0)
        numAgents = gameState.getNumAgents()
        
        # Find the best action for Pacman using expectimax
        bestAction = None
        bestScore = float('-inf')
        
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = self.expectimaxValue(successor, 0, 1, numAgents)
            if score > bestScore:
                bestScore = score
                bestAction = action
        
        return bestAction
    
    def expectimaxValue(self, gameState, depth, agentIndex, numAgents):
        """Recursively compute the expectimax value for a given state."""
        # Terminal states
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # Check if we've reached the maximum depth
        if depth == self.depth:
            return self.evaluationFunction(gameState)
        
        # Get legal actions for current agent
        legalActions = gameState.getLegalActions(agentIndex)
        
        # Handle case with no legal actions
        if not legalActions:
            return self.evaluationFunction(gameState)
        
        # Determine next agent and depth
        nextAgent = (agentIndex + 1) % numAgents
        newDepth = depth + 1 if nextAgent == 0 else depth
        
        if agentIndex == 0:
            # Pacman's turn - maximize
            bestScore = float('-inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = self.expectimaxValue(successor, newDepth, nextAgent, numAgents)
                bestScore = max(bestScore, score)
            return bestScore
        else:
            # Ghost's turn - expectimax (average of all children)
            totalScore = 0
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = self.expectimaxValue(successor, newDepth, nextAgent, numAgents)
                totalScore += score
            # Uniform probability - average of all child values
            return totalScore / len(legalActions)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    # Extract useful information from the game state
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()
    
    # Handle terminal states
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')
    
    # Calculate distance to nearest food (closer is better)
    if foodList:
        nearestFoodDist = min([manhattanDistance(pos, foodPos) for foodPos in foodList])
        foodScore = 10.0 / (nearestFoodDist + 1)
    else:
        foodScore = 100  # Big bonus for eating all food
    
    # Calculate ghost-related scores
    ghostScore = 0
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDist = manhattanDistance(pos, ghostPos)
        scaredTime = ghostState.scaredTimer
        
        if scaredTime > 0:
            # Scared ghost - chase it!
            # The closer the scared ghost, the better
            ghostScore += 200.0 / (ghostDist + 1)
        else:
            # Active ghost - avoid it!
            if ghostDist < 2:
                # Too close - major penalty
                ghostScore -= 500
            elif ghostDist < 4:
                # Getting close - moderate penalty
                ghostScore -= 50.0 / (ghostDist + 1)
            else:
                # Far away - small penalty
                ghostScore -= 5.0 / (ghostDist + 1)
    
    # Capsule score - fewer capsules is better
    capsuleScore = -20.0 * len(capsules)
    
    # Food remaining score - fewer food is better
    foodRemainingScore = -5.0 * len(foodList)
    
    # Combine all factors
    totalScore = score + foodScore + ghostScore + capsuleScore + foodRemainingScore
    
    return totalScore

# Abbreviation
better = betterEvaluationFunction
