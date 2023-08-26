from __future__ import print_function
__author__ = "Harshvardhan Gupta"
import copy


win_states = (
[0, 1, 2],
[3, 4, 5],
[6, 7, 8],
[0, 3, 6],
[1, 4, 7],
[2, 5, 8],
[0, 4, 8],
[2, 4, 6])

class Tic(object):


    def __init__(self, selfSide ,positions=[]):
        assert isinstance(positions,list) and (selfSide=="o" or
                                               selfSide=="x")
        self.side = selfSide
        if len(positions) == 0:
            self.positions = [None for i in range(9)]
        else:
            self.positions = copy.deepcopy(positions)

    def display(self):
        print("")
        for i in range(0,9):

            if i % 3 == 0:
                print("")

            if(self.positions[i]!=None):
                print(self.positions[i]," ",
                      end="")
            else:
                print ("T"," ",end="")

        print("")

    def getNextPossibleStates(self):
        possibleStates = []
        c=0
        for i in range(9):
            if self.positions[i]==None:
                possibleStates.append(Tic(self.side,self.positions))
                possibleStates[-1].positions[i] = self.side
                # possibleStates[-1].display()


        return possibleStates

    def getOppPlayer(self):
        if self.side =="o" :
            return "x"
        else:
            return "o"

    def caclulateScore(self):
        for win_state in enumerate(win_states):

            if self.positions[win_state[1][0]] == self.side and self.positions[win_state[1][1]] ==self.side and self.positions[win_state[1][2]] == self.side :
                return 1
            elif self.positions[win_state[1][0]] == self.getOppPlayer() and self.positions[win_state[1][1]] ==self.getOppPlayer() and self.positions[win_state[1][2]] == self.getOppPlayer() :
                return -1
            else :
                return 0




    def isGameOver(self):
        for i in enumerate(self.positions):
            if i[1] ==None:
                return False
        return True


def determine(tic):
    assert (isinstance(tic,Tic))

    if(tic.isGameOver()):
        return tic
    # tic.display()
    states = []
    scores = []
    count = 0

    for state in enumerate(tic.getNextPossibleStates()):
       # finalState = copy.deepcopy(determine(state[1]))
       # score = finalState.caclulateScore()
       # states.append(state[1])
       # scores.append(score)
       count += 1
    print(count)

        # if score ==1:
        #     return state[1]


'''    maxScore = -1
    maxScoreState = 0
    for score in enumerate(scores):

        if score[1] ==1:
            return states[score[0]]
        elif score[1] ==0:
            maxScore = 0
            maxScoreState = states[score[0]]

    if maxScoreState==0:
        return states[0]
    return maxScoreState
'''







#testBoard = ["x",None,None,"o","o",None,None,None,None]
testBoard = []
testTic =  Tic("x",testBoard)
testTic.display()
#print("displayed")
# testTic.display()
# print("that was init")
# print(testTic.caclulateScore())
# testTic.getNextPossibleStates()
#determine(testTic).display()

print(determine(testTic))









