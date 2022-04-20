import asyncio
import websockets
import json
import numpy as np

from uuid import uuid4

class Vis3DObject():
    def __init__(self, type, name='unnamed', uuid=None):
        self.type = type
        self.name = name
        self.uuid = uuid
        
        if uuid is None:
            self.uuid = str(uuid4())
    
    def todict(self):
        return dict(
            type=self.type,
            uuid=self.uuid,
            name=self.name
        )
    
    def tojson(self):
        return json.dumps(self.todict())
    
    def __repr__(self):
        return str(self.todict())


class PointCloud(Vis3DObject):
    def __init__(self, points, color=0xffffff, size=0.1, *args, **kwargs):
        super().__init__('PointCloud', *args, **kwargs)
        self.points = points
        self.color = color
        self.size = size
    
    def todict(self):
        obj_dict = super().todict()
        obj_dict['data'] = dict(
            points=self.points.tolist() if isinstance(self.points, np.ndarray) else self.points,
            color=self.color.tolist() if isinstance(self.color, np.ndarray) else self.color,
            size=self.size
        )
        return obj_dict


class Box(Vis3DObject):
    def __init__(self, data, color=0xffffff, opacity=0.1, linewidth=1, direction=False,
                 dashed=False, scale=1, dashsize=0.2, gapsize=0.1,
                 *args, **kwargs):
        super().__init__('Box', *args, **kwargs)
        self.bbox = data
        self.color = color
        self.opacity = opacity
        self.linewidth = linewidth
        self.direction = direction
        self.dashed = dashed
        self.scale = scale
        self.dashsize = dashsize
        self.gapsize = gapsize
    
    def todict(self):
        obj_dict = super().todict()
        obj_dict['data'] = dict(
            data=self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else self.bbox,
            color=self.color,
            opacity=self.opacity,
            linewidth=self.linewidth,
            direction=self.direction,
            dashed=self.dashed
        )
        if self.dashed:
            obj_dict['data'].update(dict(
                scale=self.scale,
                dashsize=self.dashsize,
                gapsize=self.gapsize
            ))
        return obj_dict


class Vis3DManager():
    def __init__(self, host='0.0.0.0', port=1008):
        self.host = host
        self.port = port
        self.clients = set()
        self.objects = dict()
    
    def __call__(self):
        return self.server
    
    def serve(self):
        return websockets.serve(self._serve_websocket, self.host, self.port)
    
    async def sync(self):
        if len(self.clients) > 0:
            await asyncio.wait([ self.handleSync(ws) for ws in self.clients ])
    
    async def handleSync(self, ws):
        await ws.send(json.dumps(dict(
            type='sync', 
            data={uuid: obj.todict() for uuid, obj in self.objects.items()}
        )))
    
    async def add(self, obj):
        assert(isinstance(obj, Vis3DObject))
        self.objects[obj.uuid] = obj
        if len(self.clients) > 0:
            await asyncio.wait([ws.send(json.dumps(
                dict(type='add', data=obj.todict())
            )) for ws in self.clients])
    
    async def remove(self, obj):
        assert(isinstance(obj, Vis3DObject))
        if obj.uuid not in self.objects:
            return
        del self.objects[obj.uuid]
        if len(self.clients) > 0:
            await asyncio.wait([ws.send(json.dumps(
                dict(type='remove', data=obj.uuid)
            )) for ws in self.clients])
    
    async def show(self, obj):
        assert(isinstance(obj, Vis3DObject))
        if obj.uuid not in self.objects:
            return
        if len(self.clients) > 0:
            await asyncio.wait([ws.send(json.dumps(
                dict(type='show', data=obj.uuid)
            )) for ws in self.clients])

    async def hide(self, obj):
        assert(isinstance(obj, Vis3DObject))
        if obj.uuid not in self.objects:
            return
        if len(self.clients) > 0:
            await asyncio.wait([ws.send(json.dumps(
                dict(type='hide', data=obj.uuid)
            )) for ws in self.clients])
    
    async def _serve_websocket(self, ws, path):
        try:
            self.clients.add(ws)
            async for msg in ws:
                req = json.loads(msg)
                req_type = req.get('type', None)
                req_data = req.get('data', None)
                if req_type == 'sync':
                    await self.handleSync(ws)
                else:
                    pass
        finally:
            self.clients.remove(ws)