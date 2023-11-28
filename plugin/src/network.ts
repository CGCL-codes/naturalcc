import axios from 'axios';

async function sendGetRequestWithBody(url: string, body: any): Promise<any> {
    const response = await axios.get(url, {
        data: body // Axios 允许在GET请求中使用 'data' 属性来发送body
    });

    return response.data;
}