You are a helpful assistant. Solve this puzzle for me.

There are three pegs and n disks of different sizes stacked on the first peg. The disks are
numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:

1. Only one disk can be moved at a time.
2. Each move consists of taking the upper disk from one stack and placing it on top of
   another stack.
3. A larger disk may not be placed on top of a smaller disk.

The goal is to move the entire stack to the third peg.

Example: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2, 1],
[], []], and a solution might be:

```
moves = [[1 , 0 , 2] , [2 , 0 , 1] , [1 , 2 , 1] , [3 , 0 , 2] ,
[1 , 1 , 0] , [2 , 1 , 2] , [1 , 0 , 2]]
```

This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.
Requirements:

- When exploring potential solutions in your thinking process, always include the corre-
  sponding complete list of moves.
- The positions are 0-indexed (the leftmost peg is 0).
- Ensure your final answer includes the complete list of moves in the format:
  moves = [[disk id, from peg, to peg], ...]
