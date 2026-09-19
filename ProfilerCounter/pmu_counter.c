/*
 * Copyright (c) 2021-2022 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**************************************************************************//**
 * @file     pmu_counter.c
 * @version  V1.00
 * @brief    Performance monitor counters(PMU, SysTick) functions
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <stdbool.h>
#include <inttypes.h>

#include "log_macros.h"             /* Logging macros */
#include "pmu_counter.h"
#include "ethosu_profiler.h"

#include "NuMicro.h"

//#define HAS_FREERTOS
#define CPU_PROFILE_ENABLED

static uint64_t s_u64CPUCycleCount = 0;    /* 64-bit cpu cycle counter */
static bool s_bSysTickInit = false;
extern uint32_t SystemCoreClock;        /* Expected to come from the cmsis-device lib */


static bool add_pmu_counter(uint64_t value,
                            const char *name,
                            const char *unit,
                            pmu_counters *counters)
{
    const uint32_t idx = counters->num_counters;

    if (idx < NUM_PMU_COUNTERS)
    {
        counters->counters[idx].value = value;
        counters->counters[idx].name = name;
        counters->counters[idx].unit = unit;
        ++counters->num_counters;
        return true;
    }

    printf_err("Failed to add PMU counter!\n");
    return false;
}

/**
 * Gets the current SysTick derived counter value
 *
 * The upstream version "guarded" the VAL read with NVIC_DisableIRQ(SysTick_IRQn),
 * but SysTick is a system exception (IRQn -1) and the CMSIS NVIC calls ignore
 * negative numbers - the guard compiled to nothing. The tick could then land
 * between reading VAL and reading the count, making the result one whole
 * period (10 ms) off either way. Measured on the board: 134 backward steps of
 * up to 10 ms in 20 s of polling. Every (now - t0) timeout loop that met one
 * in its first 10 ms saw a huge elapsed time and gave up at once - the
 * intermittent ESP_ERR_TIMEOUT on /generate that had nothing to do with how
 * long the backend took.
 *
 * So mask all interrupts for these few instructions, and fold in a wrap that
 * has happened but whose handler has not run yet.
 */
static uint64_t Get_SysTick_Cycle_Count(void)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();

    uint64_t count = s_u64CPUCycleCount;
    uint32_t load  = SysTick->LOAD;
    uint32_t val   = SysTick->VAL & SysTick_VAL_CURRENT_Msk;

    if (SCB->ICSR & SCB_ICSR_PENDSTSET_Msk)
    {
        /* The wrap is behind us, so a fresh VAL is from the new period -
         * unless the counter is still sitting at 0, the old period's last
         * cycle (the tick pends on reaching 0, the reload comes a clock
         * later). */
        val = SysTick->VAL & SysTick_VAL_CURRENT_Msk;
        if (val != 0u)
            count += (uint64_t)load + 1u;
    }

    __set_PRIMASK(primask);
    return count + (load - val);
}
/**
 * SysTick initialisation
 */
static int Init_SysTick(void)
{
#if !defined(HAS_FREERTOS)
    const uint32_t ticks_10ms = SystemCoreClock / 100 + 1;
    int err = 0;

    if (s_bSysTickInit)
        return 0;

    /* Reset CPU cycle count value. */
    s_u64CPUCycleCount = 0;
    /* Changing configuration for sys tick => guard from being
     * interrupted. */
    NVIC_DisableIRQ(SysTick_IRQn);
    /* SysTick init - this will enable interrupt too. */
    err = SysTick_Config(ticks_10ms);
    /* Enable interrupt again. */
    NVIC_EnableIRQ(SysTick_IRQn);

    /* Wait for SysTick to kick off */
    while (!err && !SysTick->VAL)
    {
        __NOP();
    }

    s_bSysTickInit = true;
    return err;
#else
    /* FreeRTOS has been initiated systick */
    return 0;
#endif
}

#if !defined(HAS_FREERTOS)
void SysTick_Handler(void)
{
    /* Increment the cycle counter based on load value. */
    s_u64CPUCycleCount += SysTick->LOAD + 1;
    __DSB();
    __ISB();
}
#endif

void FreeRTOS_TickHook(uint32_t u32CurrentTickCnt)
{
    s_u64CPUCycleCount = (uint64_t)u32CurrentTickCnt * (SysTick->LOAD + 1);
}


void pmu_reset_counters(void)
{
    if (0 != Init_SysTick())
    {
        printf_err("Failed to initialise system tick config\n");
        return;
    }

#if defined(ARM_NPU)
    ethosu_pmu_init();
#endif /* defined (ARM_NPU) */
    //debug("system tick config ready\n");
}

void pmu_get_counters(pmu_counters *counters)
{
    counters->num_counters = 0;
    counters->initialised = true;
    uint32_t i = 0;
#if defined (ARM_NPU)
    ethosu_pmu_counters npu_counters = ethosu_get_pmu_counters();

    for (i = 0; i < ETHOSU_PMU_NCOUNTERS; ++i)
    {
        add_pmu_counter(
            npu_counters.npu_evt_counters[i].counter_value,
            npu_counters.npu_evt_counters[i].name,
            npu_counters.npu_evt_counters[i].unit,
            counters);
    }

    for (i = 0; i < ETHOSU_DERIVED_NCOUNTERS; ++i)
    {
        add_pmu_counter(
            npu_counters.npu_derived_counters[i].counter_value,
            npu_counters.npu_derived_counters[i].name,
            npu_counters.npu_derived_counters[i].unit,
            counters);
    }

    add_pmu_counter(
        npu_counters.npu_total_ccnt,
        "NPU TOTAL",
        "cycles",
        counters);
#else  /* defined (ARM_NPU) */
    UNUSED(i);
#endif /* defined (ARM_NPU) */
#if defined(CPU_PROFILE_ENABLED)
    add_pmu_counter(
        Get_SysTick_Cycle_Count(),
        "CPU TOTAL",
        "cycles",
        counters);
#endif /* defined(CPU_PROFILE_ENABLED) */
#if !defined(CPU_PROFILE_ENABLED)
    UNUSED(Get_SysTick_Cycle_Count);
#if !defined(ARM_NPU)
    UNUSED(add_pmu_counter);
#endif /* !defined(ARM_NPU) */
#endif /* !defined(CPU_PROFILE_ENABLED) */
}

uint64_t pmu_get_systick_Count(void)
{
    return  Get_SysTick_Cycle_Count();
}

