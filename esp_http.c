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
#define HTTP_SEND_CHUNK 2048    /* AT+CIPSEND hard limit per send */

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

    raw = esp_tcp_recv_http(s_rx, HTTP_RX_SZ - 1, timeout_ms);
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
                   "Connection: keep-alive\r\n"
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
                   "Connection: keep-alive\r\n"
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

/* Where the value of header `key` starts inside [hdr, hdr+hdr_len), or -1.
 * key is lowercase and includes the colon ("content-length:"); the match is
 * case-insensitive (uvicorn sends lowercase header names). */
static int header_find(const uint8_t *hdr, int hdr_len, const char *key)
{
    const int klen = (int)strlen(key);

    for (int i = 0; i + klen <= hdr_len; i++)
    {
        int j = 0;
        while (j < klen)
        {
            char c = (char)hdr[i + j];
            if (c >= 'A' && c <= 'Z') c = (char)(c + 32);
            if (c != key[j]) break;
            j++;
        }
        if (j < klen) continue;

        int p = i + klen;
        while (p < hdr_len && (hdr[p] == ' ' || hdr[p] == '\t'))
            p++;
        return p;
    }
    return -1;
}

/* Value of the Content-Length header, or -1. */
static int header_content_length(const uint8_t *hdr, int hdr_len)
{
    int p = header_find(hdr, hdr_len, "content-length:");
    int v = 0, seen = 0;

    if (p < 0)
        return -1;
    for (; p < hdr_len && hdr[p] >= '0' && hdr[p] <= '9'; p++)
    {
        v = v * 10 + (hdr[p] - '0');
        seen = 1;
    }
    return seen ? v : -1;
}

/* Copy the value of header `key` (up to the line end) into out; "" when the
 * header is absent. */
static void header_copy(const uint8_t *hdr, int hdr_len, const char *key,
                        char *out, int cap)
{
    int p = header_find(hdr, hdr_len, key), n = 0;

    if (p >= 0)
        while (p < hdr_len && hdr[p] != '\r' && hdr[p] != '\n' && n < cap - 1)
            out[n++] = (char)hdr[p++];
    out[n] = '\0';
}

/* Shared tail of the binary requests (request already sent): receive the raw
 * response into buf, close, check the status line and Content-Length, pull
 * out header hdr_key if asked (hdr_val may be NULL), and move the body to
 * the front of buf. */
static int recv_binary(uint8_t *buf, int buf_cap, int *body_len, int timeout_ms,
                       http_hdr_out_t *hdrs, int nHdrs)
{
    int raw, status;

    for (int i = 0; i < nHdrs; i++)
        if (hdrs[i].val && hdrs[i].cap > 0)
            hdrs[i].val[0] = '\0';

    /* Headers + body land in the caller's buffer — no intermediate copy. */
    raw = esp_tcp_recv_http((char *)buf, buf_cap, timeout_ms);
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
    if (blen < 0) blen = 0;

    /* Before any verdict: the caller may want the headers of a truncated
     * response too (e.g. which answer to re-fetch). Parse before the
     * memmove below destroys the header bytes. */
    for (int i = 0; i < nHdrs; i++)
        if (hdrs[i].key && hdrs[i].val && hdrs[i].cap > 0)
            header_copy(buf, hdr_len, hdrs[i].key, hdrs[i].val, hdrs[i].cap);

    /* A dropped +IPD segment must FAIL, not pass as a short HTTP 200 —
     * a truncated JPEG written to SD looks synced forever. */
    int clen = header_content_length(buf, hdr_len);
    if (clen >= 0 && clen != blen)
    {
        /* Seen now and then on 37 KB downloads. The counters (since boot)
         * tell an MCU-side loss (ring full / UART overrun) from bytes that
         * never left the ESP. */
        printf("[HTTP] truncated: %d of %d body bytes (rx ring overflow %u, "
               "uart overrun %u)\n", blen, clen,
               (unsigned)esp_ll_rx_overflow(), (unsigned)esp_ll_rx_hw_errors());
        return HTTP_ERR_TRUNC;
    }

    if (blen > 0)
        memmove(buf, buf + hdr_len, (size_t)blen);
    *body_len = blen;

    return status;
}

int http_get_binary(const char *host, int port, const char *path,
                    uint8_t *buf, int buf_cap, int *body_len, int timeout_ms)
{
    int len, rc;

    if (!host || !path || !buf || !body_len || buf_cap < 64)
        return ESP_ERR_PARAM;
    *body_len = 0;

    len = snprintf(s_req, sizeof(s_req),
                   "GET %s HTTP/1.1\r\n"
                   "Host: %s:%d\r\n"
                   "Connection: keep-alive\r\n"
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

    return recv_binary(buf, buf_cap, body_len, timeout_ms, NULL, 0);
}

int http_post_binary(const char *host, int port, const char *path,
                     const uint8_t *body, int body_len,
                     uint8_t *buf, int buf_cap, int *resp_len, int timeout_ms,
                     http_hdr_out_t *hdrs, int nHdrs)
{
    int len, rc;

    if (!host || !path || !body || body_len <= 0 ||
        !buf || !resp_len || buf_cap < 64)
        return ESP_ERR_PARAM;
    *resp_len = 0;

    len = snprintf(s_req, sizeof(s_req),
                   "POST %s HTTP/1.1\r\n"
                   "Host: %s:%d\r\n"
                   "Content-Type: application/octet-stream\r\n"
                   "Content-Length: %d\r\n"
                   "Connection: keep-alive\r\n"
                   "\r\n",
                   path, host, port, body_len);
    if (len <= 0 || len >= (int)sizeof(s_req))
        return ESP_ERR_PARAM;

    if ((rc = esp_tcp_connect(host, port)) != ESP_OK)
        return rc;

    if ((rc = esp_tcp_send(s_req, len)) != ESP_OK)
    {
        esp_tcp_close();
        return rc;
    }

    /* The server only answers once the whole body is in, so nothing arrives
     * that esp_tcp_send's RX flush could throw away. */
    for (int off = 0; off < body_len; off += HTTP_SEND_CHUNK)
    {
        int n = body_len - off;

        if (n > HTTP_SEND_CHUNK)
            n = HTTP_SEND_CHUNK;
        if ((rc = esp_tcp_send((const char *)body + off, n)) != ESP_OK)
        {
            esp_tcp_close();
            return rc;
        }
    }

    return recv_binary(buf, buf_cap, resp_len, timeout_ms, hdrs, nHdrs);
}
