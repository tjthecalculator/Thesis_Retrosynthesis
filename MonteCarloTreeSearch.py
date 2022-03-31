from rdkit import Chem
from rdkit.Chem import AllChem
from math import sqrt
import numpy as np
import tensorflow as tf
import time
import json

class Node:

    __NodeIdx = 0

    def __init__(self, Data, Parent=None, Prior=0.0):
        Node.__NodeIdx        += 1
        self.__NodeIdx        = Node.__NodeIdx
        self.Data             = Data
        self.ParentNode       = Parent
        self.ChildNodes       = []
        self.NodeVisited      = 0
        self.ActionValue      = 0.0
        self.PriorProbability = Prior
        self.RewardScore      = 0.0

    def GetNodeIdx(self):
        return self.__NodeIdx

    def GetNodeData(self):
        return self.Data

    def GetParentNode(self):
        return self.ParentNode

    def GetChildNodes(self):
        return self.ChildNodes

    def GetNodeVisited(self):
        return self.NodeVisited

    def GetActionValue(self):
        return self.ActionValue

    def GetPriorProbability(self):
        return self.PriorProbability

    def GetRewardScore(self):
        return self.RewardScore

    def AddChildNode(self, ChildNode):
        self.ChildNodes.append(ChildNode)

    def IncreaseNodeVisited(self):
        self.NodeVisited += 1

    def SetRewardScore(self, Reward):
        self.RewardScore = Reward

    def SetActionValue(self, Action):
        self.ActionValue = Action
    

class Tree:

    def __init__(self, RootNodeData):
        self.RootNode = Node(RootNodeData)
        self.TreeNodes = [[self.RootNode]]

    def GetTreeNode(self, TreeNodeIdx):
        for i, x in enumerate(self.TreeNodes):
            for j, node in enumerate(x):
                if TreeNodeIdx == node.GetNodeIdx():
                    return i, j, node

    def AddChildNode(self, ParentNode, ChildNodeData, Prior=None):
        i, j, _   = self.GetTreeNode(ParentNode.GetNodeIdx())
        ChildNode = Node(ChildNodeData, Parent=ParentNode, Prior=Prior)
        if len(self.TreeNodes) == i + 1:
            self.TreeNodes.append([])
        self.TreeNodes[i+1].append(ChildNode)
        self.TreeNodes[i][j].AddChildNode(ChildNode)

    def IsRootNode(self, TreeNode):
        if TreeNode.GetNodeIdx() == self.RootNode.GetNodeIdx():
            return True
        else:
            return False

    def IsLeafNode(self, TreeNode):
        if not TreeNode.GetChildNodes():
            return True
        else:
            return False

    def GetRootNode(self):
        return self.RootNode

    def GetLeafNode(self):
        LeafNodes = [node for x in self.TreeNodes for node in x if self.IsLeafNode(node)]
        return LeafNodes

    def GetBranchs(self, ParentNode):
        Lv, _, _  = self.GetTreeNode(ParentNode.GetNodeIdx())
        LeafNodes = self.GetLeafNode()
        Branchs   = []
        for node in LeafNodes:
            LeafLv, _, _ = self.GetTreeNode(node.GetNodeIdx())
            if LeafLv < Lv:
                continue
            branch = [node]
            while LeafLv > Lv:
                if not self.IsRootNode(node):
                    node = node.GetParentNode()
                    branch.append(node)
                    LeafLv -= 1
            if branch[-1].GetNodeIdx() == ParentNode.GetNodeIdx():
                Branchs.append(branch)
        return Branchs

def CalculateActionValue(ParentNode, Node, Exploration=3.0):
    Q = Node.GetActionValue()/Node.GetNodeVisited()
    U = Exploration*Node.GetPriorProbability()*sqrt(ParentNode.GetNodeVisited())/(1 + Node.GetNodeVisited())
    return Q + U

def SelectChild(ParentNode):
    ChildNodes = [node for node in ParentNode.GetChildNodes()]
    Action     = [(node, CalculateActionValue(ParentNode, node)) for node in ChildNodes]
    MaxAction  = sorted(Action, key=lambda x:x[1], reverse=True)
    return MaxAction[0][0]

def UpdateActionValue(ParentNode, TreeSearch, LMax=10, Damping=0.99):
    ActionValues = []
    for node in ParentNode.GetChildNodes():
        ChildBranchs = TreeSearch.GetBranchs(node)
        Branchs      = [(LMax - (len(childs) - sum(Damping*n.GetPriorProbability() for n in childs)))/LMax for childs in ChildBranchs]
        if Branchs:
            ActionValues.append(max(Branchs)*node.GetRewardScore()*node.GetNodeVisited())
    return sum(ActionValues)/ParentNode.GetNodeVisited()

def CalculateReward(Mols, BuildingBlocks):
    MolScore = [1 if mol in BuildingBlocks else 0 for mol in Mols]
    if MolScore:
        if not sum(MolScore):
            return -1
        elif sum(MolScore)/len(MolScore) == 1:
            return 10
        else:
            return sum(MolScore)/len(MolScore)
    else:
        return -1

def Fingerprint_to_Array(Mols, Size):
    try:
        FPs  = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(m), radius=2, useChirality=True, nBits=Size) for m in Mols]
        NFPs = [np.zeros((0,)) for i in range(len(Mols))]
        for i, fp in enumerate(FPs):
            Chem.DataStructs.ConvertToNumpyArray(fp, NFPs[i])
    except ValueError:
        NFPs = [np.zeros((Size,)) for i in range(len(Mols))]
    return sum(NFPs)

def RunReaction(Reaction, Reactants):
    Mols = [Chem.MolFromSmiles(mol) for mol in Reactants]
    try:
        Outcome = AllChem.ReactionFromSmarts(Reaction).RunReactants(Mols)
    except Exception:
        Outcome = []

    Output = []
    if Outcome:
        Outcome = Outcome[0]
        for mol in Outcome:
            try:
                Chem.SanitizeMol(mol)
                Output.append(Chem.MolToSmiles(mol))
            except Exception:
                continue
        
    return Output

class MCTS:

    def __init__(self, TargetMol, BuildingBlocks=None, RetrosynRule=None, Expansion=None, Inscope=None, SearchIter=10, SimIter=10):
        self.RetrosynTree   = Tree([TargetMol])
        self.BuildingBlocks = BuildingBlocks
        self.RetrosynRule   = RetrosynRule
        self.ExpansionModel = Expansion
        self.InscopeModel   = Inscope
        self.SearchIter     = SearchIter
        self.SimIter        = SimIter
        self.ERROR          = False
        self.InitiateSearch()

    def SelectionPhase(self):
        while self.CurrentState.GetChildNodes():
            self.CurrentState = SelectChild(self.CurrentState)
            self.CurrentState.IncreaseNodeVisited()
            self.CurrentState.SetRewardScore(CalculateReward(self.CurrentState.GetNodeData(), self.BuildingBlocks))

    def ExpansionPhase(self):
        Products = [mol for mol in self.CurrentState.GetNodeData() if mol not in self.BuildingBlocks]
        if Products:
            ExpansionInput  = Fingerprint_to_Array(Products, Size=32768)
            ExpansionOutput = self.ExpansionModel.predict(ExpansionInput.reshape(1, -1))[0]
            RuleIdx         = np.argmax(ExpansionOutput)
            Prior           = ExpansionOutput[RuleIdx]
            try:
                Reactants     = RunReaction(self.RetrosynRule[RuleIdx]['Reaction_SMARTS'], Products)
                if Reactants:
                    ReactionInput = Fingerprint_to_Array(Products + Reactants, Size=2048).reshape(1, -1)
                    ProductInput  = Fingerprint_to_Array(Products, Size=16384).reshape(1, -1)
                    InscopeOutput = self.InscopeModel.predict([ProductInput, ReactionInput])
                    if InscopeOutput[0][0] >= 0.5 and Reactants != self.CurrentState.GetNodeData():
                        self.RetrosynTree.AddChildNode(self.CurrentState, Reactants, Prior=Prior)
            except Exception as e:
                print(e)
                pass

            if self.CurrentState.GetChildNodes():
                ChildNodes        = self.CurrentState.GetChildNodes()
                self.CurrentState = sorted([(node, node.GetPriorProbability()) for node in ChildNodes], key=lambda x:x[1], reverse=True)[0][0]
                self.CurrentState.IncreaseNodeVisited()
            else:
                self.ERROR = True

    def SimulationPhase(self, SimDepth=5):
        
        self.SimTree = Tree(self.CurrentState.GetNodeData())

        for _ in range(self.SimIter):
            self.SimState = self.SimTree.GetRootNode()
            self.SimState.IncreaseNodeVisited()
            for _ in range(SimDepth):
                Products = [mol for mol in self.SimState.GetNodeData() if mol not in self.BuildingBlocks]
                if Products:
                    RolloutInput  = Fingerprint_to_Array(Products, Size=32768)
                    RolloutOutput = self.ExpansionModel.predict(RolloutInput.reshape(1, -1))[0]
                    RuleIdx       = np.random.choice(len(self.RetrosynRule), p=RolloutOutput)
                    Prior         = RolloutOutput[RuleIdx]
                    try:
                        Reactants = RunReaction(self.RetrosynRule[RuleIdx]['Reaction_SMARTS'], Products)
                        if Reactants:
                            self.SimTree.AddChildNode(self.SimState, Reactants, Prior=Prior)
                    except Exception as e:
                        print(e)
                        pass
                if self.SimState.GetChildNodes():
                    self.SimState = np.random.choice(self.SimState.GetChildNodes())
                    self.SimState.IncreaseNodeVisited()
                    self.SimState.SetRewardScore(CalculateReward(self.SimState.GetNodeData(), self.BuildingBlocks))
                else:
                    break

        self.SimState = self.SimTree.GetRootNode()
        self.CurrentState.SetActionValue(UpdateActionValue(self.SimState, self.SimTree))

    def UpdatePhase(self):
        while not self.RetrosynTree.IsRootNode(self.CurrentState):
            self.CurrentState = self.CurrentState.GetParentNode()
            self.CurrentState.SetActionValue(UpdateActionValue(self.CurrentState, self.RetrosynTree))

    def InitiateSearch(self):
        Init_Time = time.time()
        for _ in range(self.SearchIter):
            self.CurrentState = self.RetrosynTree.GetRootNode()
            self.CurrentState.IncreaseNodeVisited()
            self.SelectionPhase()
            self.ExpansionPhase()
            if not self.ERROR:
                if CalculateReward(self.CurrentState.GetNodeData(), self.BuildingBlocks) >= 0.9:
                    End_Time       = time.time()
                    self.TimeUsage = End_Time - Init_Time
                    break
                else:
                    self.SimulationPhase()
                    self.UpdatePhase()
            else:
                End_Time       = time.time()
                self.TimeUsage = End_Time - Init_Time
                break
        else:
            End_Time       = time.time()
            self.TimeUsage = End_Time - Init_Time
        
    def GetResults(self):
        self.CurrentState = self.RetrosynTree.GetRootNode()
        while not self.RetrosynTree.IsLeafNode(self.CurrentState):
            self.CurrentState = SelectChild(self.CurrentState)

        Path = []
        while not self.RetrosynTree.IsRootNode(self.CurrentState):
            Path.append('.'.join(self.CurrentState.GetNodeData()))
            self.CurrentState = self.CurrentState.GetParentNode()

        Path += self.CurrentState.GetNodeData()
        SynthesisPath = ['{}>>{}'.format(Path[i], Path[i+1]) for i in range(len(Path) - 1)]
        return SynthesisPath, self.TimeUsage

