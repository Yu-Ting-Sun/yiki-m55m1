/**************************************************************************//**
 * @file     PerfTimer.h
 * @brief    Wall-clock timing for the PoC.
 *
 * Default source = ProfilerCounter/pmu_counter.c: SysTick 10 ms IRQ
 * accumulating a 64-bit CPU cycle count (proven on this exact silicon — the
 * old project timed inference/FPS with it).
 *
 * ⚠ DWT CYCCNT is DISABLED by default (PERFTIMER_TRY_DWT=0). Field finding on
 * this board: DWT is dead even with a debugger attached (CYCCNTENA never
 * sticks, CTRL reads 0x80000000), and poking DWT/CoreSight registers WITHOUT
 * a live debug session (standalone boot via the reset button) HANGS the core
 * — the boot log stopped exactly between "Target system: M55M1" and the
 * timing-source print, i.e. inside PerfTimer_Init's DWT accesses. With no
 * debugger, the debug power domain is down and those accesses lock up.
 * Only enable PERFTIMER_TRY_DWT on silicon where DWT is known-good.
 *
 * Wrap notes: 64-bit cycles → no practical wrap; the us/ms getters truncate
 * to uint32 (us wraps ~71 min) — fine for (end - start) intervals.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __PERF_TIMER_H__
#define __PERF_TIMER_H__

#include <stdint.h>
#include "NuMicro.h"       /* SystemCoreClock */
#include "pmu_counter.h"   /* pmu_reset_counters(), pmu_get_systick_Count() */

/* 0 = never touch DWT/CoreSight registers (SAFE on this board — see header).
 * 1 = try DWT CYCCNT first, fall back to SysTick if it does not count. */
#ifndef PERFTIMER_TRY_DWT
#define PERFTIMER_TRY_DWT   (0)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Initialise the timing source. Call once after BoardInit().
 * @return 1 = DWT CYCCNT active; 0 = SysTick 64-bit source active.
 */
static inline int PerfTimer_Init(void)
{
#if PERFTIMER_TRY_DWT
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    /* CoreSight Software Lock release (DWT base 0xE0001000 + LAR 0xFB0). */
    *(volatile uint32_t *)0xE0001FB0UL = 0xC5ACCE55UL;
    DWT->CYCCNT = 0U;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;

    if ((DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) != 0U)
    {
        uint32_t t0 = DWT->CYCCNT;
        for (volatile int i = 0; i < 64; i++) { }
        if (DWT->CYCCNT != t0)
            return 1;                                 /* DWT alive */
    }
    DWT->CTRL &= ~DWT_CTRL_CYCCNTENA_Msk;             /* best effort off */
#endif /* PERFTIMER_TRY_DWT */

    pmu_reset_counters();                             /* SysTick 10 ms IRQ src */
    return 0;
}

/** @brief  Current CPU cycle count. */
static inline uint64_t PerfTimer_Cycles64(void)
{
#if PERFTIMER_TRY_DWT
    if ((DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) != 0U)
        return (uint64_t)DWT->CYCCNT;                 /* 32-bit, wraps 19.5 s */
#endif
    return pmu_get_systick_Count();                   /* 64-bit, no wrap */
}

/** @brief  Current time in microseconds (uint32; wraps ~71 min — use for intervals). */
static inline uint32_t GetSystemTick_us(void)
{
    return (uint32_t)(PerfTimer_Cycles64() / (SystemCoreClock / 1000000UL));
}

/** @brief  Current time in milliseconds (uint32; use for intervals). */
static inline uint32_t GetSystemTick_ms(void)
{
    return (uint32_t)(PerfTimer_Cycles64() / (SystemCoreClock / 1000UL));
}

#ifdef __cplusplus
}
#endif

#endif /* __PERF_TIMER_H__ */
