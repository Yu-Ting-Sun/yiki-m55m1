/**************************************************************************//**
 * @file     day2_config.example.h
 * @brief    Wi-Fi / backend settings TEMPLATE.
 *
 *           Copy this file to `day2_config.h` and fill in your values —
 *           the real day2_config.h is git-ignored so credentials never
 *           enter the repository.
 *
 *   WIFI_SSID / WIFI_PWD : 2.4 GHz network only (ESP8266 has no 5 GHz).
 *                          Avoid " , or \ characters (AT escaping not handled).
 *   BACKEND_HOST         : the PC's LAN IPv4 (ipconfig), NOT 127.0.0.1.
 *   BACKEND_PORT         : uvicorn port (backend README: uvicorn main:app
 *                          --host 0.0.0.0 --port 8000).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef DAY2_CONFIG_H
#define DAY2_CONFIG_H

#define WIFI_SSID       "your-2.4ghz-ssid"
#define WIFI_PWD        "your-wifi-password"
#define BACKEND_HOST    "192.168.x.x"
#define BACKEND_PORT    8000

#endif /* DAY2_CONFIG_H */
