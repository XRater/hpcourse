package com.stupid58fly.concurrent.jcstress;

import com.stupid58fly.concurrent.LockFreePriorityQueue;
import com.stupid58fly.concurrent.PriorityQueue;
import org.openjdk.jcstress.annotations.*;
import org.openjdk.jcstress.infra.results.I_Result;

@JCStressTest
// expected value (0 + 100) * (100 - 0 + 1) = 5050
@Outcome(id = "5050", expect = Expect.ACCEPTABLE)
public class SRSWTest {
    @State
    public static class PriorityQueueState {
        public final PriorityQueue<Integer> queue = new LockFreePriorityQueue<>();
    }

    protected int sequenceCount = 100;
    protected int sum = 0;

    @Actor
    public void writer(PriorityQueueState state) {
        for (int i = 0; i < sequenceCount; i++)
            state.queue.add(i);
    }

    @Actor
    public void reader(PriorityQueueState state) {
        for (int i = 0; i < sequenceCount; ) {
            Integer value = state.queue.poll();
            if (value != null) {
                sum += value;
                i++;
            }
        }
    }

    @Arbiter
    public void arbiter(PriorityQueueState state, I_Result result) {
        result.r1 = sum;
    }
}
