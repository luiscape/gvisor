# Restore optimization

I am interested in greatly optimizing gvisor restore time. Read the restore code-path
and identify restoring operations. Use the benchmarking program to determine progress.

```
sudo ./restorebench
```

On each iteration step, build gvisor, run the bench, and store results in a file called `progress.md`. Each line needs to have:

* step number
* optimization strategy
* restore time
* improvement vs baseline

Start by collecting a baseline.

Before going ahead, outline a nubmer of optimization strategies and the estimated impact.
