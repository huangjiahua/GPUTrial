#pragma once

#ifdef linux
#define EXCHANGE        __sync_fetch_and_add
#define CAS             __sync_val_compare_and_swap
#else
#define EXCHANGE        InterlockedIncrement
#define CAS             InterlockedCompareExchange
#endif