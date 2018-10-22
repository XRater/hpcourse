
#include <pthread.h>
#include <iostream>
#include <unistd.h>
#include <random>


class Value {
public:
    Value() : _value(0) { }
 
    void update(int value) {
        _value = value;
    }
 
    int get() const {
        return _value;
    }
 
private:
    int _value;
};

pthread_mutex_t mutex;
pthread_mutex_t consumer_mutex;
pthread_cond_t ready_for_consume;
pthread_cond_t variable_prepared;

int commonConsumersSum;
//bool hasToWork = true;
bool consumeProduceFlag = true;
bool hasToStop = false;

void* producer_routine(void* arg) {
    while (true) {
        pthread_mutex_lock(&mutex);
        std::cout << "producer: mutex locked\n";
        // Wait for consumer to start
        while(consumeProduceFlag) {
            pthread_cond_wait(&ready_for_consume, &mutex);
        }
        std::cout << "producer asked\n";
        // Read data, loop through each value and update the value, notify consumer, wait for consumer to process
        int buffer;
        if (!(std::cin >> buffer)) {
            std::cout << "producer: reading from buffer failed\n";
            //hasToWork = false;
            pthread_mutex_unlock(&mutex);
            break;
        }
        std::cout << "buffer was red" << std::endl;
        static_cast<Value*>(arg)->update(buffer);
        consumeProduceFlag = true;
        pthread_cond_signal(&variable_prepared);
        pthread_mutex_unlock(&mutex);
    }
    return nullptr;
}

int maxSleepMs;
volatile int consum_num = 0;
void* consumer_routine(void* arg) {
    int tnum = ++consum_num;
    while (true) {
    // notify about start
        pthread_mutex_lock(&consumer_mutex);

        pthread_mutex_lock(&mutex);
        std::cout << tnum << " consumer: mutex locked\n";
        consumeProduceFlag = false;
        pthread_cond_signal(&ready_for_consume);
        std::cout << tnum << " consumer: notified producer\n";
    // for every update issued by producer, read the value and add to sum
        while (!hasToStop && !consumeProduceFlag) {
            pthread_cond_wait(&variable_prepared, &mutex);
        }
        if (hasToStop) {
            std::cout << tnum << " has to stop\n";
            pthread_mutex_unlock(&mutex);
            pthread_mutex_unlock(&consumer_mutex);
            break;
        }
        std::cout << "notified about update" << std::endl;
        commonConsumersSum += static_cast<Value*>(arg)->get();
        pthread_mutex_unlock(&mutex);

        pthread_mutex_unlock(&consumer_mutex);
        usleep((rand() % maxSleepMs)*1000);
    }
    // return pointer to result (aggregated result for all consumers)
    return &commonConsumersSum;
}
 
void* consumer_interruptor_routine(void* arg) {
    // wait for consumer to start
 
    // interrupt consumer while producer is running  
    return nullptr;                                        
}
 
int run_threads(int nThreads) {
    Value v;
    // start N threads and wait until they're done
    // return aggregated sum of values
    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_init(&consumer_mutex, NULL);
    pthread_cond_init(&ready_for_consume, NULL);
    pthread_cond_init(&variable_prepared, NULL);
  
    pthread_t threads[nThreads];
    pthread_t prodThread;
    
    pthread_create(&prodThread, NULL, producer_routine, &v);
    for (int i = 0; i < nThreads; i++) {
        pthread_create(&threads[i], NULL, consumer_routine, &v);
    }
    
    pthread_join(prodThread, nullptr);
    std::cout << "run_threads: joined producer\n";
    hasToStop = true;
    pthread_cond_signal(&variable_prepared);
    /*
    for (int i = 0; i < nThreads; i++) {
        pthread_cancel(threads[i]);
    }
    std::cout << "all consumers cancelled\n";
    */
    void * resPointer = nullptr;
    pthread_join(threads[0], &resPointer);
    for (int i = 1; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_cond_destroy(&variable_prepared);
    pthread_cond_destroy(&ready_for_consume);
    pthread_mutex_destroy(&consumer_mutex);
    pthread_mutex_destroy(&mutex);
    

    return *(int*)resPointer;
}
 
int main(int argc, char* argv[]) {
    int numThreads = atoi(argv[1]);
    
    maxSleepMs = atoi(argv[2]);
    
    std::cout << "num threads: " << numThreads << std::endl;
    std::cout << run_threads(numThreads) << std::endl;
    return 0;
}
