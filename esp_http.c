/**************************************************************************//**
 * @file     esp_http.c
 * @brief    Minimal HTTP/1.1 client over esp_at TCP (see esp_http.h).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "esp_at.h"
#include "esp_http.h"

#define HTTP_REQ_SZ     1024    /* request line + headers + JSON body */
#define HTTP_RX_SZ      4096    /* raw response: headers + body */
#define HTTP_RECV_MS    5000    /* whole-response deadline */

static char s_req[HTTP_REQ_SZ];
static char s_rx[HTTP_RX_SZ];

/* Parse status code + copy body out of a raw HTTP response. */
static int parse_response(int raw_len, char *response, int max_len)
{
    int   status;
    char *body;

    s_rx[raw_len] = '\0';
    response[0]   = '\0';

    if (strncmp(s_rx, "HTTP/1.", 7) != 0 || raw_len < 12)
        return HTTP_ERR_PARSE;

    status = atoi(&s_rx[9]);               /* "HTTP/1.1 200 OK" -> 200 */
    if (status < 100 || status > 599)
        return HTTP_ERR_PARSE;

    body = strstr(s_rx, "\r\n\r\n");
    if (body)
    {
        int body_len = raw_len - (int)(body + 4 - s_rx);
        if (body_len > max_len - 1)
            body_len = max_len - 1;
        if (body_len > 0)
            memcpy(response, body + 4, (size_t)body_len);
        response[body_len > 0 ? body_len : 0] = '\0';
    }

    return status;
}

/* Common path: connect -> send prebuilt s_req -> recv -> close -> parse. */
static int do_request(const char *host, int port, int req_len,
                      char *response, int max_len, int timeout_ms)
{
    int rc, raw;

    if (!response || max_len <= 0)
        return ESP_ERR_PARAM;
    if (req_len <= 0 || req_len >= HTTP_REQ_SZ)
        return ESP_ERR_PARAM;              /* also catches snprintf truncation */

    if ((rc = esp_tcp_connect(host, port)) != ESP_OK)
        return rc;

    if ((rc = esp_tcp_send(s_req, req_len)) != ESP_OK)
    {
        esp_tcp_close();
        return rc;
    }

    raw = esp_tcp_recv(s_rx, HTTP_RX_SZ - 1, timeout_ms);
    esp_tcp_close();                       /* no-op if server already closed */

    if (raw < 0)
        return raw;

    return parse_response(raw, response, max_len);
}

int http_get(const char *host, int port, const char *path,
             char *response, int max_len)
{
    int len;

    if (!host || !path)
        return ESP_ERR_PARAM;

    len = snprintf(s_req, sizeof(s_req),
                   "GET %s HTTP/1.1\r\n"
                   "Host: %s:%d\r\n"
                   "Connection: close\r\n"
                   "\r\n",
                   path, host, port);

    return do_request(host, port, len, response, max_len, HTTP_RECV_MS);
}

int http_post_json(const char *host, int port, const char *path,
                   const char *json_body,
                   char *response, int max_len)
{
    return http_post_json_ex(host, port, path, json_body,
                             response, max_len, HTTP_RECV_MS);
}

int http_post_json_ex(const char *host, int port, const char *path,
                      const char *json_body,
                      char *response, int max_len, int timeout_ms)
{
    int len;

    if (!host || !path || !json_body)
        return ESP_ERR_PARAM;

    len = snprintf(s_req, sizeof(s_req),
                   "POST %s HTTP/1.1\r\n"
                   "Host: %s:%d\r\n"
                   "Content-Type: application/json\r\n"
                   "Content-Length: %d\r\n"
                   "Connection: close\r\n"
                   "\r\n"
                   "%s",
                   path, host, port, (int)strlen(json_body), json_body);

    return do_request(host, port, len, response, max_len, timeout_ms);
}

/* Binary-safe needle search (the body may contain any byte values). */
static const uint8_t *memfind(const uint8_t *hay, int hay_len, const char *needle)
{
    int nlen = (int)strlen(needle);

    for (int i = 0; i + nlen <= hay_len; i++)
        if (memcmp(hay + i, needle, (size_t)nlen) == 0)
            return hay + i;
    return 0;
}

int http_get_binary(const char *host, int port, const char *path,
                    uint8_t *buf, int buf_cap, int *body_len, int timeout_ms)
{
    int len, rc, raw, status;

    if (!host || !path || !buf || !body_len || buf_cap < 64)
        return ESP_ERR_PARAM;
    *body_len = 0;

    len = snprintf(s_req, sizeof(s_req),
                   "GET %s HTTP/1.1\r\n"
                   "Host: %s:%d\r\n"
                   "Connection: close\r\n"
                   "\r\n",
                   path, host, port);
    if (len <= 0 || len >= (int)sizeof(s_req))
        return ESP_ERR_PARAM;

    if ((rc = esp_tcp_connect(host, port)) != ESP_OK)
        return rc;

    if ((rc = esp_tcp_send(s_req, len)) != ESP_OK)
    {
        esp_tcp_close();
        return rc;
    }

    /* Headers + body land in the caller's buffer — no intermediate copy. */
    raw = esp_tcp_recv((char *)buf, buf_cap, timeout_ms);
    esp_tcp_close();

    if (raw < 0)
        return raw;
    if (raw < 12 || memcmp(buf, "HTTP/1.", 7) != 0)
        return HTTP_ERR_PARSE;

    status = (buf[9] - '0') * 100 + (buf[10] - '0') * 10 + (buf[11] - '0');
    if (status < 100 || status > 599)
        return HTTP_ERR_PARSE;

    const uint8_t *sep = memfind(buf, raw, "\r\n\r\n");
    if (!sep)
        return HTTP_ERR_PARSE;

    int hdr_len = (int)(sep + 4 - buf);
    int blen    = raw - hdr_len;

    if (blen > 0)
        memmove(buf, buf + hdr_len, (size_t)blen);
    *body_len = (blen > 0) ? blen : 0;

    return status;
}
