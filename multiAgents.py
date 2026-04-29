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
        """
        1. see the future
        2. gather information
        3. start with the base score 
        4. closer to food -> bigger reward
        5. closer to ghost -> BAD
        6. penalize lots of remaining food
        7. penalize stopping
        """

        # Useful information you can extract from a GameState (pacman.py)

        # -------------------------------------------------------
        # 1. sucessor state: game state AFTER pacman takes 'action'
        # -------------------------------------------------------
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # -------------------------------------------------------
        # 2. info from successor state
        # -------------------------------------------------------

        newPos = successorGameState.getPacmanPosition()     # (x, y)
        newFood = successorGameState.getFood()  # Ture: food at the pos

        newGhostStates = successorGameState.getGhostStates()

        # scaredTimer > 0 : ghost is scared
        # scaredTimer == 0: ghost is dangerous
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # -------------------------------------------------------
        # 3. built in score
        #   - Eating food:            +10
        #   - Eating last food/win:   +500
        #   - Dying:                  -500
        #   - Eating scared ghost:    +200
        #   - Every move penalty:     -1
        #   - Scared duration:         40
        # -------------------------------------------------------
        score = successorGameState.getScore()

        # -------------------------------------------------------
        # 4. reward being closer to food
        #   1 / distance:
        #       dist = 1 -> +1.0    (close, big reward)
        #       dist = 5 -> +0.2
        #       dist =10 -> +0.1    (far, small reward)
        # -------------------------------------------------------

        foodList = newFood.asList() # list of (x,y) pos with food

        if foodList:
            # manhattan distance = |x2 - x1| + |y2 - y1|
            nearestFood = min(
                manhattanDistance(newPos, foodPos) for foodPos in foodList
            )
            if nearestFood > 0:
                score += 1.0 / nearestFood


        # -------------------------------------------------------
        # 5. penalize being close to ghost,
        #   bur reward if they are scared ghost
        #   
        #   - DANGER: scaredTimer == 0 
        #   penalize before dying to steer pacman away
        #  
        #   - scared ghost: scaredTimer > 0
        #   eating scared ghost = +200, move toward them!
        # -------------------------------------------------------

        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):

            ghostPos = ghostState.getPosition()
            dist     = manhattanDistance(newPos, ghostPos)

            if scaredTime == 0:
                # collision occurs at dist <= 0.7
                
                if dist <= 1:
                    score -= 500    # basically too late to survive
                elif dist <= 2:
                    score -= 100    # very close
                else:
                    score -= 2.0 / dist # be aware but no need to be scared yet

            else:
                # ghost is SCARED 
                # only chase if scared time is enough to reach it
                if dist > 0 and scaredTime > dist:
                    score += 200.0 / dist   # bigger reward when close
                elif dist > 0:
                    score == 1.0 / dist     # smaller reward even if it's risky

        # -------------------------------------------------------
        # 6. penalize if food leftover 
        #  
        #   fewer food remaining = win bonus
        #   encourage eating as much as food
        # -------------------------------------------------------

        score -= 4 * len(foodList)

        # -------------------------------------------------------
        # 7. penalize stopping
        # -------------------------------------------------------
        
        if action == Directions.STOP:
            score -= 50

        return score

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
        # start minimax at Pacman (agent 0) with full depth
        # returns (score, action) -> score needed for search
        score, action = self.minimax(gameState, agentIndex=0, depth=self.depth)
        return action
    
    def minimax(self, gameState, agentIndex, depth):
        """
        decide which case applies at this node.
        returns (value, action)

        four cases:
            1. terminal state:  eval immediately
            2. depth exhausted: eval immediately
            3. pacmans' turn:   maxValue
            4. ghost's turn:    minValue
        """

        # case 1: terminal node, no action to return
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # case 2: depth exhausted. only check when agenetIndex == 0
        if agentIndex == 0 and depth == 0:
            return self.evaluationFunction(gameState), None

        # case 3: pacman's turn. MAX node. (pacman's agentIndex == 0)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        
        # case 4: ghost's turn. MIN node.
        else:
            return self.minValue(gameState, agentIndex, depth)
        
    def maxValue(self, gameState, agentIndex, depth):
        """
        pacman's turn:
            v = -inf
            for each successor of state:
                v = max(v, value(successor))
            return v
        Returns (bestScore, bestAction)
        """
        
        bestScore = float('-inf')
        bestAction = None
        
        # agentIndex = (agentIndex + 1) % numAgents
        nextAgent = 1

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)

            # depth stays the same until all ghosts move
            score, _ = self.minimax(successor, nextAgent, depth)

            if score > bestScore:
                bestScore = score
                bestAction = action
    
        return bestScore, bestAction

    def minValue(self, gameState, agentIndex, depth):
        """
        ghost's turn:
            v = +inf
            for each successor of state:
                v = min(v, value(successor))
            return v

        multiple min layers (one for each ghost) for each max layer

        Returns (bestScore, bestAction)
        """

        bestScore = float('+inf')
        bestAction = None

        numAgents = gameState.getNumAgents()
        lastGhost = (agentIndex == numAgents - 1)

        if lastGhost:
            # back to packman
            nextAgent = 0
            nextDepth = depth - 1
        else:
            nextAgent = agentIndex + 1
            nextDepth = depth
        
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)

            score, _ = self.minimax(successor, nextAgent, nextDepth)

            if score < bestScore:
                bestScore  = score
                bestAction = action

        return bestScore, bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        Alpha: MAX's best option (lower bound for MAX)
        Beta:  MIN's best option (upper bound for MIN)

        """
        # start alpha-beta at Pacman (agent0) with full depth
        # alpha = -inf (MAX's init best), beta = +inf (MIN's init best)
        score, action = self.alphaBeta(
            gameState,
            agentIndex=0,
            depth=self.depth,
            alpha=float('-inf'),
            beta=float('+inf')
        )
        return action
    
    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        """
        Decide which case applies at this node.
        Returns (value, action)

        Four cases (mirrors minimax):
            1. terminal state:  eval immediately
            2. depth exhausted: eval immediately (checked only when agentIndex == 0)
            3. pacman's turn:   maxValue
            4. ghost's turn:    minValue
        """
        # case 1: terminal node, no action to return
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # case 2: depth exhausted; only check when back at Pacman
        if agentIndex == 0 and depth == 0:
            return self.evaluationFunction(gameState), None

        # case 3: pacman's turn. MAX node
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)

        # case 4: ghost's turn. MIN node
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        """
        Pacman's turn (MAX node):
            v = -inf
            for each successor of state:
                v = max(v, value(successor))
                if v > beta: prune (return early) <- MIN above will never choose this
                alpha = max(alpha, v)
            return v

        Returns (bestScore, bestAction)
        """
        bestScore = float('-inf')
        bestAction = None

        nextAgent = 1

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)

            score, _ = self.alphaBeta(successor, nextAgent, depth, alpha, beta)

            if score > bestScore:
                bestScore = score
                bestAction = action

            # prune: MIN ancestor already has a better option, stop here.
            # does NOT prune on equality
            if bestScore > beta:
                return bestScore, bestAction

            # update alpha
            alpha = max(alpha, bestScore)

        return bestScore, bestAction

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        """
        Ghost's turn (MIN node):
            v = +inf
            for each successor of state:
                v = min(v, value(successor))
                if v < alpha: prune (return early) <- MAX above will never choose this
                beta = min(beta, v)
            return v

        Multiple min layers (one per ghost) per max layer

        Returns (bestScore, bestAction)
        """
        bestScore = float('+inf')
        bestAction = None

        numAgents = gameState.getNumAgents()
        lastGhost = (agentIndex == numAgents - 1)

        if lastGhost:
            # back to Pacman
            nextAgent = 0
            nextDepth = depth - 1
        else:
            nextAgent = agentIndex + 1
            nextDepth = depth

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)

            score, _ = self.alphaBeta(successor, nextAgent, nextDepth, alpha, beta)

            if score < bestScore:
                bestScore = score
                bestAction = action

            # prune: MAX ancestor already has a better option, stop here
            # does NOT prune on equality
            if bestScore < alpha:
                return bestScore, bestAction

            # update beta: best MIN value seen on this path to root
            beta = min(beta, bestScore)

        return bestScore, bestAction

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
        # start at Pacman (agent0) with full depth
        score, action = self.expectimax(
            gameState,
            agentIndex=0,
            depth=self.depth
        )

        return action
    
    def expectimax(self, gameState, agentIndex, depth):
        """
        Decide which case applies at this node.
        Returns (value, action)

        Four cases:
            1. terminal state:  eval immediately
            2. depth exhausted: eval immediately (checked only when agentIndex == 0)
            3. pacman's turn:   maxValue
            4. ghost's turn:    expValue  (expected value over uniform random actions)
        """
        # case 1: terminal node, no action to return
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # case 2: depth exhausted; only check when back at Pacman
        if agentIndex == 0 and depth == 0:
            return self.evaluationFunction(gameState), None

        # case 3: pacman's turn. MAX node
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)

        # case 4: ghost's turn. CHANCE node (uniform random)
        else:
            return self.expValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        # identical logic as MinimaxAgent.maxValue 
        # but recurses into self.expectimax

        bestScore = float('-inf')
        bestAction = None
    
        nextAgent = 1

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)

            score, _ = self.expectimax(successor, nextAgent, depth)

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestScore, bestAction


    def expValue(self, gameState, agentIndex, depth):
        """
        Ghost's turn (CHANCE node):
            Ghosts are chosen uniformly at random from legal actions
            v = 0
            p = 1 / len(legalActions) <- (uniform probability)
            for each successor of state:
                v += p * value(successor)
            return v

        No bestAction to track: chance nodes don't "choose", so we return None

        Returns (expectedScore, None)
        """
        numAgents = gameState.getNumAgents()
        lastGhost = (agentIndex == numAgents - 1)

        if lastGhost:
            nextAgent = 0
            nextDepth = depth - 1
        else:
            nextAgent = agentIndex + 1
            nextDepth = depth

        legalActions = gameState.getLegalActions(agentIndex)

        # uniform probability over all legal ghost actions
        prob = 1.0 / len(legalActions)

        expectedScore = 0.0

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)

            score, _ = self.expectimax(successor, nextAgent, nextDepth)

            # accumulate weighted contribution of each branch
            expectedScore += prob * score

        # chance nodes have NO single best action to return
        return expectedScore, None
    

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
        linear combination of weighted features (5)

        features:
            1. current score        : cummulative baseline
            2. closest foods        : reciprocal pulls pacman toward foods
            3. remaining foods      : penalize uneaten food
            4. remaining pellets    : penalized uneaten pellets
            5. proximity of ghost   : continuous reciprocal penalty for active ghosts,
                                      reward chasing scared ghost, but not when they are about to be un-scrared
    """
    
    pos         = currentGameState.getPacmanPosition()
    foodList    = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules    = currentGameState.getCapsules()
    
    # -------------------------------------------------------
    # 1. curr score (base) 
    # -------------------------------------------------------
    score = currentGameState.getScore()
    
    # -------------------------------------------------------
    # 2. closest foods: reciprocal dist to closest food
    # -------------------------------------------------------
    if foodList:
        closestFood = min(manhattanDistance(pos, food) for food in foodList)
        score += 1.0 / closestFood

    # -------------------------------------------------------
    # 3. remaining food penalty (-4 / food)
    #   encourages pacman to get clear all food
    # -------------------------------------------------------  
    score -= 4.0 * len(foodList)

    # -------------------------------------------------------
    # 4. remaining pellets (capsule) penalty (-4 / pellet)
    # -------------------------------------------------------
    score -= 4.0 * len(capsules)
    
    # -------------------------------------------------------
    # 5. proximity of ghost
    #   continuous reciprocal, magnitude scales with proximity
    #   e.g., ghost at dist = 3 should be worse than dist = 7
    # -------------------------------------------------------

    for ghost in ghostStates:
        dist = manhattanDistance(pos, ghost.getPosition())

        if ghost.scaredTimer > 0:
            # scared ghost: reward chasing, but scale by scaredTimer 
            # so pacman doesn't chase a ghost about to unscared
            # scaredTimer is normalize to ~[0,1]
            timerWeight = ghost.scaredTimer / 40.0
            score += timerWeight * (2.0 / (dist + 1))
        else:
            # active ghost: continuous penalty, the closer, the worse.
            # -10 / (dist + 1) -> dist = 0 -> -10, dist = 4 -> -2
            # add a hard cliff at dist <= 1 
            score -= 10.0 / (dist + 1)
            if dist <= 1:
                score -= 500.0   # RIP instant death

    return score

# Abbreviation
better = betterEvaluationFunction
