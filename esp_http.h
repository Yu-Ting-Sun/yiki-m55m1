/**************************************************************************//**
 * @file     esp_http.h
 * @brief    Day-2 Task 2: minimal HTTP/1.1 client over the esp_at TCP layer.
 *
 * One request = one TCP connection ("Connection: close"), so the ESP's
 * "CLOSED" marker doubles as end-of-response — no chunked parsing needed
 * (uvicorn answers small JSON with Content-Length, never chunked).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef ESP_HTTP_H
#define ESP_HTTP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HTTP_ERR_PARSE   (-10)  /* got bytes, but no valid HTTP status line */

/* All functions return the HTTP status code (e.g. 200) on success, or a
 * negative ESP_ERR_* / HTTP_ERR_* code. For the text variants the response
 * BODY (headers stripped) is copied into `response` and NUL-terminated
 * (truncated to max_len-1). */
int http_get(const char *host, int port, const char *path,
             char *response, int max_len);

int http_post_json(const char *host, int port, const char *path,
                   const char *json_body,
                   char *response, int max_len);

/* Same as http_post_json but with a caller-chosen whole-response deadline —
 * needed for /generate, where the LLM keeps the connection waiting 10-30 s. */
int http_post_json_ex(const char *host, int port, const char *path,
                      const char *json_body,
                      char *response, int max_len, int timeout_ms);

/* Binary-safe GET for large payloads (Day-3 story image). The raw response
 * (headers + body) is received straight into `buf` (so buf must hold BOTH:
 * payload + ~512 B of headers), then the body is moved to the front of
 * `buf`; *body_len gets its length. No NUL termination. */
int http_get_binary(const char *host, int port, const char *path,
                    uint8_t *buf, int buf_cap, int *body_len, int timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* ESP_HTTP_H */
